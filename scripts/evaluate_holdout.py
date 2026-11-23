"""F4.3 — Evaluación ÚNICA en HOLDOUT (2023-07-01…2024-06-30) con modelo y umbrales congelados en DEV.

Lee configs/seleccion.yaml (no lo modifica). Evalúa la configuración elegida y todas las líneas base,
cada una con SU umbral de DEV. Escribe reports/holdout.md y reports/holdout_metricas.csv.
Requiere que existan las ejecuciones en reports/runs/holdout/ (scripts/run_eval.py --period holdout).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from metro_anomaly import data as D
from metro_anomaly.calendar import holiday_name
from metro_anomaly.evaluation import block_bootstrap, metrics, paired_recall_diff
from metro_anomaly.labels import heuristic_labels

ROOT = Path(__file__).resolve().parents[1]
CRIT = {"D1": (0.02, 0.10), "D2_severa": 0.95, "D2_moderada": 0.80, "D3": 0.10, "D4": 0.95}


def load(period, det, agg, seed=0):
    with open(ROOT / "reports" / "runs" / period / f"{det}__{agg}__s{seed}.pkl", "rb") as fh:
        return pickle.load(fh)


def main():
    sel = yaml.safe_load(open(ROOT / "configs" / "seleccion.yaml"))
    chosen = f"{sel['detector']}/{sel['agg']}"
    df, _ = D.load_all(ROOT / "data")
    W = D.line_series(df, "LINEA A", "2022-01-01", "2024-06-30")
    lab = {"resid_k1.5": heuristic_labels(W, "resid")["any"], "raw_k1.5": heuristic_labels(W, "raw")["any"]}

    rows, res_by = [], {}
    for cfg, thr in sel["baselines_umbral"].items():
        det, agg = cfg.split("/")
        seeds = sorted(int(p.stem.split("__s")[1]) for p in (ROOT / "reports/runs/holdout").glob(f"{det}__{agg}__s*.pkl"))
        if not seeds:
            continue
        per_seed = []
        for s in seeds:
            r = load("holdout", det, agg, s)
            m = metrics(r, thr)
            for name, y in lab.items():
                yy = y.reindex(r.real.index)
                ok = yy.notna() & r.real["score"].notna()
                m[f"auc_vs_{name}"] = roc_auc_score(yy[ok], r.real["score"][ok])
            per_seed.append(m)
            if s == seeds[0]:
                res_by[cfg] = (r, thr)
        m = pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()
        m.update(config=cfg, umbral_dev=thr, n_semillas=len(seeds), elegido=(cfg == chosen))
        rows.append(m)
    M = pd.DataFrame(rows).set_index("config")
    M.to_csv(ROOT / "reports" / "holdout_metricas.csv")

    r, thr = res_by[chosen]
    ci = block_bootstrap(r, thr, ["tasa_alertas", "tasa_alertas_festivos", "recall_severa", "recall_moderada",
                                  "recall_T5_perfil_festivo", "recall_moderada_media"], n_boot=2000)
    m = M.loc[chosen]
    checks = {
        "D1 tasa de alertas ∈ [2 %, 10 %]": (m["tasa_alertas"], CRIT["D1"][0] <= m["tasa_alertas"] <= CRIT["D1"][1]),
        "D2 recall severas ≥ 0,95": (m["recall_severa"], m["recall_severa"] >= CRIT["D2_severa"]),
        "D2 recall moderadas ≥ 0,80": (m["recall_moderada"], m["recall_moderada"] >= CRIT["D2_moderada"]),
        "D3 alertas en festivos ≤ 10 %": (m["tasa_alertas_festivos"], m["tasa_alertas_festivos"] <= CRIT["D3"]),
        "D4 recall T5 ≥ 0,95": (m["recall_T5_perfil_festivo"], m["recall_T5_perfil_festivo"] >= CRIT["D4"]),
    }
    diffs = {}
    for cfg, (rb, thb) in res_by.items():
        if cfg != chosen:
            diffs[cfg] = paired_recall_diff(r, thr, rb, thb, "moderada", n_boot=2000)
    best_other = max((c for c in diffs if not c.startswith("ae_original")), key=lambda c: M.loc[c, "recall_moderada"], default=None)
    d5a = diffs.get(best_other, {"ic_inf": np.nan})
    d5b = next((v for c, v in diffs.items() if c.startswith("ae_original")), {"ic_inf": np.nan})
    checks["D5a no peor que la mejor línea base (IC sup. de la dif. ≥ 0)"] = (d5a.get("diferencia"), d5a.get("ic_sup", -1) >= 0)
    checks["D5b supera al AE original (IC inf. de la dif. > 0)"] = (d5b.get("diferencia"), d5b.get("ic_inf", -1) > 0)

    alerts = r.real[r.real["score"] > thr].copy()
    alerts["festivo"] = [holiday_name(d) or "" for d in alerts.index]
    if r.z_real is not None:
        z = r.z_real.loc[alerts.index]
        alerts["horas_|z|>umbral"] = [", ".join(f"{c[1:]}h({v:+.1f})" for c, v in row.items() if abs(v) > thr) for _, row in z.iterrows()]
    alerts["dia"] = alerts.index.day_name()

    cols = ["umbral_dev", "n_semillas", "tasa_alertas", "tasa_alertas_festivos", "recall_severa", "recall_moderada", "recall_leve",
            "recall_moderada_media", "recall_T4_cero", "recall_T4_duplicada", "recall_T4_desfase", "recall_T5_perfil_festivo",
            "localizacion_hora", "auc_vs_resid_k1.5", "auc_vs_raw_k1.5"]
    md = ["# Evaluación en HOLDOUT (generado por `scripts/evaluate_holdout.py`)", "",
          f"Periodo: 2023-07-01…2024-06-30 · configuración elegida en DEV: **{chosen}** · umbral congelado: {thr:.4f}", "",
          "## Criterios pre-registrados (configuración elegida)", "", "| Criterio | Valor | Cumple |", "|---|---|---|",
          *[f"| {k} | {v:.3f} | {'✔' if ok else '✘'} |" if v is not None and not pd.isna(v) else f"| {k} | n/d | {'✔' if ok else '✘'} |" for k, (v, ok) in checks.items()],
          "", "IC 95 % (bootstrap por semanas, 2000 réplicas):", "", ci.round(3).to_markdown(), "",
          "## Todas las configuraciones (cada una con su umbral de DEV)", "", M[cols].round(3).to_markdown(), "",
          "## Diferencia pareada de recall moderado (elegido − otro)", "",
          pd.DataFrame(diffs).T.round(3).to_markdown(), "",
          f"## Alertas del modelo elegido en días reales del holdout ({len(alerts)})", "",
          alerts.drop(columns=["mes"], errors="ignore").round(2).to_markdown(), ""]
    (ROOT / "reports" / "holdout.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
