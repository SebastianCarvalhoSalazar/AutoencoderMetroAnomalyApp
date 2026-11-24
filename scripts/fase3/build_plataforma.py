"""F10.4 — Construye los artefactos de la plataforma multilínea (reproducción cronológica causal).

Salida en artifacts/plataforma/:
- horas.csv.gz: observado (real), esperado y z (estimación) por línea × día × hora, 2022-04…2024-06.
- dias.csv.gz: evaluación por línea y día del motor de reglas (categoría, score, umbral, alcance).
- pronostico_holdout.csv.gz: pronóstico fuera de muestra del híbrido elegido en DEV (2023-07…2024-06).
- pronostico_siguiente.csv: pronóstico del primer día posterior a los datos (predicción sin dato observado).
- manifiesto.json: procedencia (SHA-256 de los datos, commit), umbrales, selección por línea, catálogo de modelos.
Uso: python scripts/fase3/build_plataforma.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import plataforma as PL
from metro_anomaly.pronostico import FEATURES_BASE, FEATURES_V2, GBMForecaster, build_rows, profile_predict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "plataforma"
REP = ROOT / "reports" / "fase3"
START, END = "2022-04-01", "2024-06-30"


def forecast_artifacts(red, selection):
    P = pd.read_pickle(REP / "pronostico_predicciones_holdout.pkl")
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp", ROOT / "scripts" / "fase3" / "exp_pronostico.py")
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
    P = exp.conformal(P.dropna(subset=["naive", "perfil", "gbm"]))
    P["modelo"] = P["linea"].map(selection)
    P["pred"] = [P.at[i, m] for i, m in zip(P.index, P["modelo"])]
    # El intervalo calibrado se construye alrededor del GBM; se re-centra en la predicción elegida (en log).
    lp, lsel = np.log1p(P["gbm"]), np.log1p(P["pred"])
    P["p10"] = np.expm1(np.log1p(P["gbm_p10c"]) - lp + lsel)
    P["p90"] = np.expm1(np.log1p(P["gbm_p90c"]) - lp + lsel)
    first = P["fecha"] < "2023-08-01"
    P.loc[first, ["p10", "p90"]] = np.nan  # primer mes sin calibración: no se muestra intervalo
    hold = P[["fecha", "linea", "hora", "y", "pred", "p10", "p90", "modelo", "naive", "gbm"]].rename(columns={"y": "observado"})
    ok = hold["p10"].notna()
    cov = hold[ok].groupby("linea").apply(lambda g: round(float(((g["observado"] >= g["p10"]) & (g["observado"] <= g["p90"])).mean()), 3),
                                          include_groups=False).to_dict()
    # Día siguiente al último dato: ajuste con todo lo disponible.
    last = max(W.index.max() for W in red.Ws.values())
    nxt = last + pd.Timedelta(days=1)
    R = build_rows(red.Ws, pd.date_range("2022-01-01", nxt), include_future=True)
    tr, te = R[R["fecha"] <= last], R[R["fecha"] == nxt].copy()
    for name, feats in (("gbm", FEATURES_BASE), ("gbm_v2", FEATURES_V2)):
        p = GBMForecaster(feats).fit(tr).predict(te)
        te[name], te[f"{name}_p10"], te[f"{name}_p90"] = p["pred"], p["p10"], p["p90"]
    prof = profile_predict(red.Ws, last, [nxt])
    te["perfil"] = [prof[ln].loc[nxt, f"h{h:02d}"] if ln in prof else np.nan for ln, h in zip(te["linea"], te["hora"])]
    te["combo"] = np.expm1((np.log1p(te["gbm"]) + np.log1p(te["perfil"])) / 2)
    te["combo_v2"] = np.expm1((np.log1p(te["gbm_v2"]) + np.log1p(te["perfil"])) / 2)
    c_last = P.groupby("linea")["c_conformal"].last()
    te["modelo"] = te["linea"].map(selection)
    te = te[te["modelo"].notna()]
    te["pred"] = [te.at[i, m] for i, m in zip(te.index, te["modelo"])]
    c = te["linea"].map(c_last).fillna(1.0)
    lg, l10, l90, ls = np.log1p(te["gbm"]), np.log1p(te["gbm_p10"]), np.log1p(te["gbm_p90"]), np.log1p(te["pred"])
    te["p10"] = np.expm1(ls - c * (lg - l10))
    te["p90"] = np.expm1(ls + c * (l90 - lg))
    # H6 (pre-registrado con el GBM base): total diario pronosticado frente al p95 de los 365 días previos.
    tot = te.groupby("linea")["gbm"].sum()
    p95 = {ln: red.Ws[ln].sum(axis=1, min_count=19).loc[last - pd.Timedelta(days=364): last].quantile(0.95) for ln in tot.index}
    te["ratio_p95_gbm"] = te["linea"].map(lambda ln: tot[ln] / p95[ln])
    nxt_df = te[["fecha", "linea", "hora", "pred", "p10", "p90", "modelo", "ratio_p95_gbm"]]
    return hold.drop(columns="gbm"), nxt_df, cov


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    red = PL.load_network(ROOT / "data")
    horas, dias = PL.build_history(red, START, END)
    thr = PL.thresholds_from_dev(dias)
    ev = PL.evaluate_days(red, horas, dias, thr)
    dias = dias.merge(ev, on=["fecha", "linea"])
    horas.to_csv(OUT / "horas.csv.gz", index=False, float_format="%.4g")
    dias.to_csv(OUT / "dias.csv.gz", index=False, float_format="%.5g")
    sel = json.loads((REP / "pronostico_holdout.json").read_text(encoding="utf-8"))["seleccion_por_linea"]
    hold, nxt, cov = forecast_artifacts(red, sel)
    hold.to_csv(OUT / "pronostico_holdout.csv.gz", index=False, float_format="%.6g")
    nxt.to_csv(OUT / "pronostico_siguiente.csv", index=False, float_format="%.6g")
    data_files = sorted((ROOT / "data").glob("*.xlsx")) + [ROOT / "data" / "lineas.csv", ROOT / "data" / "topologia_red.csv"]
    PL.write_manifest(OUT, ROOT, data_files, {"periodo": [START, END], "umbrales": thr, "seleccion_pronostico": sel,
                                               "lineas_coherencia": list(PL.COHERENCE_LINES),
                                               "cobertura_intervalo_seleccion_2023_08_2024_06": cov,
                                               "lineas_h6_validadas": PL.H6_LINES,
                                               "dias_por_categoria": dias["categoria"].value_counts().to_dict(),
                                               "segundos_construccion": round(time.time() - t0, 1)})
    print(dias["categoria"].value_counts(), thr, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
