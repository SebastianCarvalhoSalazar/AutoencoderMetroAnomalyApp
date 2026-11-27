"""P1 — Corrección AR(1) por persistencia del error del pronóstico (diseño pre-registrado en
docs/investigacion_pronostico/PLAN.md; no cambiar sin registrar la desviación en docs/BITACORA.md).

Entrada: predicciones fuera de muestra de scripts/fase3/exp_pronostico.py (reports/fase3/pronostico_predicciones_{dev,holdout}.pkl).
Salida: reports/investigacion_pronostico/p1.json y p1.md
Uso: python scripts/investigacion_pronostico/p1_persistencia.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score

from metro_anomaly import data as D
from metro_anomaly.pronostico import wape

ROOT = Path(__file__).resolve().parents[2]
F3 = ROOT / "reports" / "fase3"
OUT = ROOT / "reports" / "investigacion_pronostico"
MODELS_HOLDOUT = ["naive", "perfil", "gbm", "gbm_v2", "combo", "combo_v2", "gbm_clima"]  # filas comunes del informe vigente
MODELS_DEV = ["naive", "perfil", "gbm", "gbm_v2", "combo", "combo_v2"]
CLIP, MIN_HOURS, MIN_DAYS = 0.3, 10, 60
N_BOOT, SEED = 1000, 0


def closures() -> set:
    c = pd.read_csv(ROOT / "data" / "cierres_programados.csv")
    return {(r.linea, d) for r in c.itertuples() for d in pd.date_range(r.desde, r.hasta)}


def daily_error(P: pd.DataFrame) -> pd.DataFrame:
    """ē(t) por línea: media de log1p(y) − log1p(b) en las horas con dato (mínimo MIN_HOURS)."""
    ok = P.dropna(subset=["y", "base"])
    e = np.log1p(ok["y"]) - np.log1p(ok["base"])
    g = e.groupby([ok["linea"], ok["fecha"]])
    d = pd.DataFrame({"ebar": g.mean(), "n": g.size()}).reset_index()
    d.loc[d["n"] < MIN_HOURS, "ebar"] = np.nan
    return d[["linea", "fecha", "ebar"]]


def predictor(E: pd.DataFrame, cierres: set) -> pd.DataFrame:
    """x(t) = clip(ē(t−1), ±CLIP); 0 si no hay ē(t−1) o si t−1 fue cierre programado de la línea."""
    out = []
    for ln, g in E.groupby("linea"):
        s = g.set_index("fecha")["ebar"]
        idx = pd.date_range(s.index.min(), s.index.max() + pd.Timedelta(days=1))
        prev = s.reindex(idx).shift(1)
        x = prev.clip(-CLIP, CLIP).fillna(0.0)
        x[[(ln, d - pd.Timedelta(days=1)) in cierres for d in idx]] = 0.0
        out.append(pd.DataFrame({"linea": ln, "fecha": idx, "x": x.to_numpy(), "ebar": s.reindex(idx).to_numpy()}))
    return pd.concat(out, ignore_index=True)


def phi_table(X: pd.DataFrame, months) -> pd.DataFrame:
    """φ por línea y mes M: Huber sin intercepto de ē(t) sobre x(t) con los días de meses < M; recortado a [0, 1]."""
    rows = []
    for ln, g in X.groupby("linea"):
        g = g.dropna(subset=["ebar"])
        for m in months:
            tr = g[g["fecha"] < m]
            phi = 0.0
            if len(tr) >= MIN_DAYS:
                hr = HuberRegressor(fit_intercept=False).fit(tr[["x"]].to_numpy(), tr["ebar"].to_numpy())
                phi = float(np.clip(hr.coef_[0], 0.0, 1.0))
            rows.append({"linea": ln, "mes": m, "phi": phi, "dias": len(tr)})
    return pd.DataFrame(rows)


def apply(P: pd.DataFrame, X: pd.DataFrame, PHI: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    P["mes"] = P["fecha"].dt.to_period("M").dt.to_timestamp()
    P = P.merge(X[["linea", "fecha", "x"]], on=["linea", "fecha"], how="left").merge(PHI[["linea", "mes", "phi"]], on=["linea", "mes"], how="left")
    P["x"], P["phi"] = P["x"].fillna(0.0), P["phi"].fillna(0.0)
    P["ar"] = np.expm1(np.log1p(P["base"]) + P["phi"] * P["x"])
    return P


def r_with_ci(P: pd.DataFrame) -> dict:
    a_b, a_c = (P["y"] - P["base"]).abs(), (P["y"] - P["ar"]).abs()
    wk = P["fecha"].dt.to_period("W")
    G = pd.DataFrame({"b": a_b, "c": a_c, "w": wk}).groupby("w")[["b", "c"]].sum()
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(G), len(G))
        s = G.iloc[pick].sum()
        boots.append(1 - s["c"] / s["b"])
    return {"wape_base": round(wape(P["y"], P["base"]), 5), "wape_corregido": round(wape(P["y"], P["ar"]), 5),
            "r": float(1 - a_c.sum() / a_b.sum()), "ic_inf": float(np.percentile(boots, 2.5)), "ic_sup": float(np.percentile(boots, 97.5)),
            "semanas": len(G), "filas": len(P)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((F3 / "pronostico_holdout.json").read_text(encoding="utf-8"))
    selection, ref = sel["seleccion_por_linea"], sel
    dev, hold = (pd.read_pickle(F3 / f"pronostico_predicciones_{p}.pkl") for p in ("dev", "holdout"))
    for P in (dev, hold):
        P["base"] = [P.at[i, selection[ln]] if ln in selection else np.nan for i, ln in zip(P.index, P["linea"])]

    # 0. Comprobación previa: reproducir el WAPE vigente (seleccion_dev) del informe
    H = hold.dropna(subset=MODELS_HOLDOUT)
    rep_glob = round(wape(H["y"], H["base"]), 4)
    rep_lin = {ln: round(wape(g["y"], g["base"]), 4) for ln, g in H.groupby("linea")}
    ok_rep = (len(H) == ref["n_filas_comunes"] and rep_glob == ref["wape_global"]["seleccion_dev"]
              and all(rep_lin[ln] == ref["wape_por_linea"][ln]["seleccion_dev"] for ln in rep_lin))
    print("reproducción:", ok_rep, len(H), rep_glob)
    if not ok_rep:
        raise SystemExit("No se reproduce el WAPE vigente: se detiene (criterio del pre-registro).")

    cierres = closures()
    ALL = pd.concat([dev, hold], ignore_index=True)
    X = predictor(daily_error(ALL), cierres)
    months_h = list(pd.date_range("2023-07-01", "2024-06-01", freq="MS"))
    months_d = list(pd.date_range("2023-01-01", "2023-06-01", freq="MS"))
    PHI = phi_table(X, months_d + months_h)

    Hc = apply(H, X, PHI)
    Dl = dev.dropna(subset=MODELS_DEV)
    Dl = apply(Dl[Dl["fecha"] >= "2023-01-01"], X, PHI)
    res_h, res_d = r_with_ci(Hc), r_with_ci(Dl)
    by_line = {ln: {"wape_base": round(wape(g["y"], g["base"]), 5), "wape_corregido": round(wape(g["y"], g["ar"]), 5),
                    "r": round(1 - wape(g["y"], g["ar"]) / wape(g["y"], g["base"]), 4)} for ln, g in Hc.groupby("linea")}
    worst = min(v["r"] for v in by_line.values())
    crit = {"c1_r_ge_2pct": res_h["r"] >= 0.02, "c2_ic_inf_gt_0": res_h["ic_inf"] > 0,
            "c3_ninguna_linea_peor_1pct": all(v["wape_corregido"] <= 1.01 * v["wape_base"] for v in by_line.values()),
            "c4_replica_dev_r_gt_0": res_d["r"] > 0}
    verdict = "CUMPLE" if all(crit.values()) else "NO CUMPLE"

    # Descriptivo
    post = Hc["post_festivo"] == 1 if "post_festivo" in Hc else pd.Series(False, index=Hc.index)
    desc_post = {"dias_post_festivo": {"wape_base": round(wape(Hc.loc[post, "y"], Hc.loc[post, "base"]), 4),
                                       "wape_corregido": round(wape(Hc.loc[post, "y"], Hc.loc[post, "ar"]), 4), "filas": int(post.sum())}}
    df, _ = D.load_all(ROOT / "data")
    h6 = {}
    Dd = Hc.groupby(["linea", "fecha"])[["y", "base", "ar"]].sum().reset_index()
    for ln, g in Dd.groupby("linea"):
        tot = D.line_series(df, ln)[D.SCORED_HOURS].sum(axis=1, min_count=19)
        thr = pd.Series({f: tot.loc[f - pd.Timedelta(days=365): f - pd.Timedelta(days=1)].quantile(0.95) for f in g["fecha"]})
        gi = g.set_index("fecha")
        lab = (gi["y"] > thr).astype(int)
        if 3 <= lab.sum() < len(lab):
            h6[ln] = {"positivos": int(lab.sum()), "auc_base": round(roc_auc_score(lab, gi["base"] / thr), 3),
                      "auc_corregido": round(roc_auc_score(lab, gi["ar"] / thr), 3)}
    phi_h = PHI[PHI["mes"].isin(months_h)].pivot(index="linea", columns="mes", values="phi")
    phi_h.columns = [c.strftime("%Y-%m") for c in phi_h.columns]

    out = {"veredicto": verdict, "criterios": crit, "holdout": res_h, "replica_dev_2023_01_06": res_d, "por_linea_holdout": by_line,
           "peor_linea_r": worst, "reproduccion_wape_vigente": {"ok": ok_rep, "wape_global": rep_glob, "filas": len(H)},
           "descriptivo": {**desc_post, "H6_alta_demanda": h6, "phi_por_linea_mes": phi_h.round(3).to_dict(orient="index")},
           "seleccion_por_linea": selection, "diseno": "docs/investigacion_pronostico/PLAN.md"}
    (OUT / "p1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    pct = lambda v: f"{100 * v:+.2f} %"
    md = ["# P1 — Corrección AR(1) por persistencia del error del pronóstico (generado por `scripts/investigacion_pronostico/p1_persistencia.py`)", "",
          f"Diseño y criterio pre-registrados en `docs/investigacion_pronostico/PLAN.md`. **Veredicto: {verdict}.**", "",
          f"Reproducción del WAPE vigente: {ok_rep} ({rep_glob}, {len(H)} filas).", "",
          "| Periodo | WAPE vigente | WAPE corregido | r | IC 95 % | semanas |", "|---|---|---|---|---|---|",
          f"| Holdout 2023-07…2024-06 | {res_h['wape_base']:.4f} | {res_h['wape_corregido']:.4f} | {pct(res_h['r'])} | [{pct(res_h['ic_inf'])}; {pct(res_h['ic_sup'])}] | {res_h['semanas']} |",
          f"| Réplica DEV 2023-01…2023-06 | {res_d['wape_base']:.4f} | {res_d['wape_corregido']:.4f} | {pct(res_d['r'])} | [{pct(res_d['ic_inf'])}; {pct(res_d['ic_sup'])}] | {res_d['semanas']} |", "",
          "Criterios: " + ", ".join(f"{k} = {'sí' if v else 'no'}" for k, v in crit.items()), "",
          "## Por línea (holdout)", "", pd.DataFrame(by_line).T.to_markdown(), "",
          "## Descriptivo", "", f"Días posteriores a festivo: {desc_post['dias_post_festivo']}", "",
          "H6 (días > p95):", "", pd.DataFrame(h6).T.to_markdown(), "", "φ por línea y mes (holdout):", "", phi_h.round(3).to_markdown(), ""]
    (OUT / "p1.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
