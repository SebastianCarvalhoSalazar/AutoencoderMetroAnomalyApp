"""F10.3 — Experimento de pronóstico (H5, H5b, H6) con origen rodante mensual.

Evaluación: 2023-07…2024-06; para cada mes M, los modelos se ajustan con datos de 2022-01-01 a M-1.
Salida: reports/fase3/pronostico.md y .json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from metro_anomaly import data as D
from metro_anomaly.fuentes import clima
from metro_anomaly.pronostico import FEATURES_BASE, FEATURES_CLIMA, FEATURES_V2, GBMForecaster, build_rows, profile_predict, wape
from metro_anomaly.pronostico import conformal_online as conformal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "fase3"
LINES_EXCLUDED = {"LINEA L"}  # no opera todas las franjas (sin días completos)


PERIODS = {"holdout": ("2023-07-01", "2024-06-01"), "dev": ("2022-07-01", "2023-06-01")}
OOS_COLS = ["fecha", "linea", "hora", "y", "naive", "perfil", "gbm", "gbm_p10", "gbm_p90", "gbm_v2", "combo", "combo_v2"]
CANDIDATES = ["perfil", "gbm", "gbm_v2", "combo", "combo_v2"]  # regla de selección por línea (pre-registrada): menor WAPE en DEV


def main(period="holdout"):
    OUT.mkdir(parents=True, exist_ok=True)
    df, _ = D.load_all(ROOT / "data")
    Ws = {ln: D.line_series(df, ln)[D.SCORED_HOURS] for ln in sorted(df["line"].unique()) if ln not in LINES_EXCLUDED}
    wx = clima.load_hourly(ROOT / "data" / "clima_open_meteo_medellin.csv")
    dates = pd.date_range("2022-01-01", PERIODS[period][1][:8] + "30")
    R = build_rows(Ws, dates, wx)
    preds = []
    for ms in pd.date_range(*PERIODS[period], freq="MS"):
        me = ms + pd.offsets.MonthEnd(0)
        tr = R[R["fecha"] < ms]
        te = R[(R["fecha"] >= ms) & (R["fecha"] <= me)].copy()
        g0 = GBMForecaster(FEATURES_BASE).fit(tr)
        p0 = g0.predict(te)
        te["gbm"], te["gbm_p10"], te["gbm_p90"] = p0["pred"], p0["p10"], p0["p90"]
        if period == "holdout":  # H5b (clima oráculo) solo se evalúa en el holdout
            te["gbm_clima"] = GBMForecaster(FEATURES_BASE + FEATURES_CLIMA).fit(tr).predict(te)["pred"]
        p2 = GBMForecaster(FEATURES_V2).fit(tr).predict(te)
        te["gbm_v2"], te["gbm_v2_p10"], te["gbm_v2_p90"] = p2["pred"], p2["p10"], p2["p90"]
        te["naive"] = np.expm1(te["lag_mismo_tipo"] + te["nivel"])
        prof = profile_predict(Ws, ms - pd.Timedelta(days=1), pd.date_range(ms, me))
        te["perfil"] = [prof[ln].loc[f, f"h{h:02d}"] if ln in prof and f in prof[ln].index else np.nan
                        for ln, f, h in zip(te["linea"], te["fecha"], te["hora"])]
        te["combo"] = np.expm1((np.log1p(te["gbm"]) + np.log1p(te["perfil"])) / 2)
        te["combo_v2"] = np.expm1((np.log1p(te["gbm_v2"]) + np.log1p(te["perfil"])) / 2)
        preds.append(te)
        print(ms.date(), "ok", flush=True)
    P = pd.concat(preds)
    P["y"] = np.expm1(P["y_log"])
    P = P.dropna(subset=["y"])
    P.to_pickle(OUT / f"pronostico_predicciones_{period}.pkl")
    if period == "holdout":  # versión compacta y versionada: insumo de scripts/fase3/build_plataforma.py
        P[OOS_COLS].to_csv(OUT / "pronostico_oos_holdout.csv.gz", index=False, float_format="%.6g")
    return P


def evaluate(P: pd.DataFrame, Ws: dict, period: str, selection: dict | None = None):
    models = [m for m in ["naive", "perfil", "gbm", "gbm_v2", "combo", "combo_v2", "gbm_clima"] if m in P.columns]
    n0 = len(P)
    P = P.dropna(subset=models)  # comparación justa: solo filas con predicción de todos los modelos
    P = conformal(P)
    if selection:  # híbrido: el modelo elegido en DEV para cada línea
        P["seleccion_dev"] = [P.at[i, selection[ln]] for i, ln in zip(P.index, P["linea"])]
        models.append("seleccion_dev")
    by_line = {ln: {m: round(wape(g["y"], g[m]), 4) for m in models} for ln, g in P.groupby("linea")}
    overall = {m: round(wape(P["y"], P[m]), 4) for m in models}

    def _cov(g, lo, hi):
        return round(float(((g["y"] >= g[lo]) & (g["y"] <= g[hi])).mean()), 3)
    first = P["fecha"] < P["fecha"].min() + pd.offsets.MonthBegin(1)
    cov = {ln: _cov(g, "gbm_p10", "gbm_p90") for ln, g in P.groupby("linea")}
    covc = {ln: _cov(g, "gbm_p10c", "gbm_p90c") for ln, g in P[~first].groupby("linea")}
    width = {ln: round(float(((g["gbm_p90c"] - g["gbm_p10c"]) / g["gbm"]).median()), 3) for ln, g in P[~first].groupby("linea")}
    res = {"periodo": period, "n_filas_total": n0, "n_filas_comunes": int(len(P)), "wape_global": overall, "wape_por_linea": by_line,
           "cobertura_intervalo_80": cov, "cobertura_80_calibrada_desde_mes_2": covc, "ancho_relativo_mediano_calibrado": width,
           "seleccion_por_linea": selection}
    t = pd.DataFrame(by_line).T
    for m in ["gbm", "gbm_v2", "seleccion_dev"]:
        if m in t:
            t[f"mejora_{m}_vs_naive_%"] = (100 * (1 - t[m] / t["naive"])).round(1)
    if "gbm_clima" in t:
        t["mejora_clima_%"] = (100 * (1 - t["gbm_clima"] / t["gbm"])).round(1)
    t["cobertura_80"] = pd.Series(cov)
    t["cobertura_80_calibrada"] = pd.Series(covc)
    md = [f"# Pronóstico día siguiente por línea y franja — periodo {period} (generado por `scripts/fase3/exp_pronostico.py`)", "",
          f"Origen rodante mensual {PERIODS[period][0][:7]}…{PERIODS[period][1][:7]}; ajuste con 2022-01…M-1. WAPE = Σ|y−ŷ| / Σy (menor es mejor). "
          f"Métricas sobre las {len(P)} filas (línea × día × franja) con predicción de todos los modelos (de {n0}).", "",
          f"WAPE global: {overall}", "", t.to_markdown(), "",
          "`cobertura_80_calibrada`: banda recalibrada en línea con los meses anteriores (desde el 2.º mes; ajuste añadido después de ver la cobertura del holdout v1).", ""]
    if period == "holdout":
        rainy = P["lluvia_dia"] >= P.drop_duplicates("fecha")["lluvia_dia"].quantile(0.9)
        by_rain = {m: {"lluviosos": round(wape(P.loc[rainy, "y"], P.loc[rainy, m]), 4), "resto": round(wape(P.loc[~rainy, "y"], P.loc[~rainy, m]), 4)} for m in models}
        rd = P.loc[rainy].drop_duplicates("fecha")["fecha"].dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
        res["wape_lluvia"], res["dias_lluviosos_por_mes"] = by_rain, rd
        md += ["## Días lluviosos (lluvia del día ≥ p90)", "", pd.DataFrame(by_rain).T.to_markdown(), "",
               f"Días lluviosos por mes (confusión con el mes): {rd}", ""]
        D_ = P.groupby(["linea", "fecha"])[["y", "naive", "gbm"] + (["seleccion_dev"] if selection else [])].sum().reset_index()
        h6 = {}
        for ln, g in D_.groupby("linea"):
            tot = Ws[ln].sum(axis=1, min_count=19)
            thr = pd.Series({f: tot.loc[f - pd.Timedelta(days=365): f - pd.Timedelta(days=1)].quantile(0.95) for f in g["fecha"]})
            gi = g.set_index("fecha")
            lab = (gi["y"] > thr).astype(int)
            if lab.sum() < 3 or lab.sum() == len(lab):
                continue
            h6[ln] = {"positivos": int(lab.sum()), "dias": int(len(lab)), "auc_gbm": round(roc_auc_score(lab, gi["gbm"] / thr), 3),
                      "auc_naive": round(roc_auc_score(lab, gi["naive"] / thr), 3)}
            if selection:
                h6[ln]["auc_seleccion_dev"] = round(roc_auc_score(lab, gi["seleccion_dev"] / thr), 3)
        res["H6_alta_demanda"] = h6
        md += ["## H6 — días de alta demanda (> p95 de los 365 días previos)", "", pd.DataFrame(h6).T.to_markdown(), ""]
    (OUT / f"pronostico_{period}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / f"pronostico_{period}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return P, res


def select_per_line(res_dev: dict) -> dict:
    """Regla pre-registrada: para cada línea, el candidato con menor WAPE en DEV."""
    return {ln: min(CANDIDATES, key=lambda m: v[m]) for ln, v in res_dev["wape_por_linea"].items()}


if __name__ == "__main__":
    import sys
    df, _ = D.load_all(ROOT / "data")
    Ws = {ln: D.line_series(df, ln)[D.SCORED_HOURS] for ln in sorted(df["line"].unique()) if ln not in LINES_EXCLUDED}
    out = {}
    for per in ("dev", "holdout"):
        pkl = OUT / f"pronostico_predicciones_{per}.pkl"
        out[per] = pd.read_pickle(pkl) if ("--metricas" in sys.argv and pkl.exists()) else main(per)
    _, rdev = evaluate(out["dev"], Ws, "dev")
    evaluate(out["holdout"], Ws, "holdout", selection=select_per_line(rdev))
