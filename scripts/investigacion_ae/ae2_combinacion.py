"""AE.2 — Hipótesis A1: detector complementario (perfil contextual + autoencoder con calendario).

Regla (fijada antes de calcular; docs/BITACORA.md): score combinado = max(s_perfil/u_perfil, s_AE/u_AE) con los umbrales
de DEV de AE.1; umbral combinado = p95 del combinado en días reales de DEV (mismo presupuesto del 5 %).

Criterio pre-registrado (docs/investigacion_autoencoders/PLAN.md, A1):
  (1) recall T4 medio ≥ +0,05 frente al perfil solo con IC 95 % > 0;
  (2) recall moderado T1–T3 sin pérdida (IC del cambio ∋ 0 o > 0).

Salida: reports/investigacion_ae/ae2.json y ae2.md. Usa las corridas de AE.1 (reports/investigacion_ae/runs, semilla 0).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly.evaluation import RollingResult, budget_threshold, metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "investigacion_ae"
RUNS = OUT / "runs"
T4 = ["T4_cero", "T4_duplicada", "T4_desfase"]
MOD = ["T1_moderada", "T2_moderada", "T3_moderada"]
N_BOOT = 2000


def load(per, name):
    with open(RUNS / per / f"{name}__s0.pkl", "rb") as fh:
        return pickle.load(fh)


def fuse(pr: RollingResult, ae: RollingResult, u_pr: float, u_ae: float) -> RollingResult:
    real = pr.real[["score", "tipo_dia", "festivo", "mes"]].copy()
    a = ae.real["score"].reindex(real.index)
    rp, ra = real["score"] / u_pr, a / u_ae
    real["score"] = np.fmax(rp, ra)
    real["origen"] = np.where(rp >= ra, "perfil", "autoencoder")
    key = ["fecha", "escenario"]
    m = pr.injected.merge(ae.injected[key + ["score", "top_hora_ok"]], on=key, suffixes=("_pr", "_ae"), validate="one_to_one")
    rp, ra = m["score_pr"] / u_pr, m["score_ae"] / u_ae
    inj = m[["fecha", "escenario", "tipo", "severidad", "aplicable"]].copy()
    inj["score"] = np.fmax(rp, ra)
    inj["top_hora_ok"] = np.where(rp >= ra, m["top_hora_ok_pr"], m["top_hora_ok_ae"])
    inj["origen"] = np.where(rp >= ra, "perfil", "autoencoder")
    return RollingResult("combinado", real, inj)


def paired(a: RollingResult, ta: float, b: RollingResult, tb: float, scen: list, mean_of_scen: bool, seed=0):
    """Diferencia pareada (a − b) del recall sobre las mismas inyecciones de ``scen``; IC por bootstrap de semanas.
    ``mean_of_scen``: promedio de los recalls por escenario (recall T4 medio) en vez del recall agrupado."""
    key = ["fecha", "escenario"]
    ia = a.injected[a.injected.escenario.isin(scen) & a.injected.aplicable][key + ["score"]]
    ib = b.injected[b.injected.escenario.isin(scen) & b.injected.aplicable][key + ["score"]]
    m = ia.merge(ib, on=key, suffixes=("_a", "_b"))
    m["da"], m["db"] = (m["score_a"] > ta).astype(float), (m["score_b"] > tb).astype(float)
    m["wk"] = pd.DatetimeIndex(m["fecha"]).to_period("W")

    def stat(df):
        if mean_of_scen:
            g = df.groupby("escenario")
            return float(g["da"].mean().mean() - g["db"].mean().mean())
        return float(df["da"].mean() - df["db"].mean())
    point = stat(m)
    weeks = np.array(sorted(m["wk"].unique()))
    groups = {w: m[m["wk"] == w] for w in weeks}
    rng = np.random.default_rng(seed)
    boots = [stat(pd.concat([groups[w] for w in rng.choice(weeks, len(weeks))])) for _ in range(N_BOOT)]
    return {"diferencia": round(point, 4), "ic_inf": round(float(np.percentile(boots, 2.5)), 4),
            "ic_sup": round(float(np.percentile(boots, 97.5)), 4), "n": int(len(m))}


def main():
    ae1 = json.loads((OUT / "ae1.json").read_text(encoding="utf-8"))["configs"]
    u_pr, u_ae = ae1["perfil_contextual/peak_shift"]["umbral_dev"], ae1["ae_contextual/top2"]["umbral_dev"]
    res = {}
    for per in ("dev", "holdout"):
        res[per] = {"perfil": load(per, "perfil_contextual__peak_shift"), "ae": load(per, "ae_contextual__top2")}
        res[per]["combinado"] = fuse(res[per]["perfil"], res[per]["ae"], u_pr, u_ae)
    u_c = budget_threshold(res["dev"]["combinado"].real["score"])
    thr = {"perfil": u_pr, "ae": u_ae, "combinado": u_c}
    out = {"umbral_combinado_dev": round(u_c, 4), "regla": "max(s_perfil/u_perfil, s_AE/u_AE) > p95 de DEV", "modelos": {}}
    keep = ["tasa_alertas", "tasa_alertas_festivos", "recall_moderada", "recall_severa", "recall_leve", "localizacion_hora",
            *[f"recall_{s}" for s in MOD + T4], "recall_T1_severa", "recall_T2_severa", "recall_T3_severa", "recall_T5_perfil_festivo"]
    for per in ("dev", "holdout"):
        for k in ("perfil", "ae", "combinado"):
            m = metrics(res[per][k], thr[k])
            row = {kk: (round(float(m[kk]), 4) if kk in m and m[kk] == m[kk] else None) for kk in keep}
            row["recall_T4_medio"] = round(float(np.mean([m[f"recall_{s}"] for s in T4])), 4)
            out["modelos"][f"{per}|{k}"] = row
    h = res["holdout"]
    c1 = paired(h["combinado"], u_c, h["perfil"], u_pr, T4, mean_of_scen=True)
    c2 = paired(h["combinado"], u_c, h["perfil"], u_pr, MOD, mean_of_scen=False)
    d_dev1 = paired(res["dev"]["combinado"], u_c, res["dev"]["perfil"], u_pr, T4, mean_of_scen=True)
    d_dev2 = paired(res["dev"]["combinado"], u_c, res["dev"]["perfil"], u_pr, MOD, mean_of_scen=False)
    ok1 = c1["diferencia"] >= 0.05 and c1["ic_inf"] > 0
    ok2 = c2["ic_inf"] <= 0 <= c2["ic_sup"] or c2["ic_inf"] > 0
    out["criterio"] = {"recall_T4_medio": {**c1, "exigido": "≥ +0,05 con IC > 0", "cumple": bool(ok1)},
                       "recall_moderado_T1_T3": {**c2, "exigido": "IC ∋ 0 o > 0 (sin pérdida)", "cumple": bool(ok2)},
                       "veredicto": "CUMPLE" if ok1 and ok2 else "NO CUMPLE",
                       "dev_referencia": {"recall_T4_medio": d_dev1, "recall_moderado_T1_T3": d_dev2}}
    # quién decide en el combinado (días reales con alerta e inyecciones detectadas)
    hr = h["combinado"].real
    al = hr[hr["score"] > u_c]
    hi = h["combinado"].injected
    det = hi[hi["aplicable"] & (hi["score"] > u_c)]
    out["origen_alertas_reales"] = al["origen"].value_counts().to_dict()
    out["origen_detecciones_por_tipo"] = det.groupby("tipo")["origen"].value_counts().unstack(fill_value=0).to_dict(orient="index")
    out["alertas_reales_combinado"] = [{"fecha": str(d.date()), "origen": o, "score_rel": round(float(s) / u_c, 3)} for d, o, s in zip(al.index, al["origen"], al["score"])]
    (OUT / "ae2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    T = pd.DataFrame(out["modelos"]).T
    md = ["# AE.2 — Hipótesis A1: perfil + autoencoder (generado por `scripts/investigacion_ae/ae2_combinacion.py`)", "",
          f"Regla: {out['regla']}; umbral combinado (DEV) = {u_c:.4f}.", "", T.to_markdown(), "",
          "## Criterio pre-registrado (holdout, combinado − perfil)", "",
          f"- Recall T4 medio: {c1} → {'cumple' if ok1 else 'no cumple'} (exigido ≥ +0,05 con IC > 0)",
          f"- Recall moderado T1–T3: {c2} → {'cumple' if ok2 else 'no cumple'} (exigido IC ∋ 0 o > 0)",
          f"- **Veredicto: {out['criterio']['veredicto']}**", "",
          f"Referencia en DEV (no decide): T4 {d_dev1}; moderado {d_dev2}", "",
          f"Origen de las alertas reales del combinado en el holdout: {out['origen_alertas_reales']}", "",
          f"Origen de las detecciones por tipo: {out['origen_detecciones_por_tipo']}"]
    (OUT / "ae2.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
