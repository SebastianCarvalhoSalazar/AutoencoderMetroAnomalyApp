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
from metro_anomaly.pronostico import conformal_factor, conformal_online, forecast_day

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "plataforma"
REP = ROOT / "reports" / "fase3"
START, END = "2022-04-01", "2024-06-30"
GZ = {"method": "gzip", "mtime": 0}  # gzip sin fecha: el mismo contenido produce el mismo archivo


def forecast_artifacts(red, selection):
    # Predicciones fuera de muestra del holdout (generadas y versionadas por scripts/fase3/exp_pronostico.py).
    P = pd.read_csv(REP / "pronostico_oos_holdout.csv.gz", parse_dates=["fecha"])
    P = conformal_online(P.dropna(subset=["naive", "perfil", "gbm"]))
    P["modelo"] = P["linea"].map(selection)
    P["pred"] = [P.at[i, m] for i, m in zip(P.index, P["modelo"])]
    # El intervalo calibrado se construye alrededor del GBM; se re-centra en la predicción elegida (en log).
    lp, lsel = np.log1p(P["gbm"]), np.log1p(P["pred"])
    P["p10"] = np.expm1(np.log1p(P["gbm_p10c"]) - lp + lsel)
    P["p90"] = np.expm1(np.log1p(P["gbm_p90c"]) - lp + lsel)
    first = P["fecha"] < "2023-08-01"
    P.loc[first, ["p10", "p90"]] = np.nan  # primer mes sin calibración: no se muestra intervalo
    hold = P[["fecha", "linea", "hora", "y", "pred", "p10", "p90", "modelo", "naive"]].rename(columns={"y": "observado"})
    ok = hold["p10"].notna()
    cov = hold[ok].groupby("linea").apply(lambda g: round(float(((g["observado"] >= g["p10"]) & (g["observado"] <= g["p90"])).mean()), 3),
                                          include_groups=False).to_dict()
    # Factor conformal para pronósticos nuevos: todas las predicciones fuera de muestra disponibles de cada línea.
    c_next = {ln: conformal_factor(g["y"], g["gbm"], g["gbm_p10"], g["gbm_p90"]) for ln, g in P.groupby("linea")}
    last = max(W.index.max() for W in red.Ws.values())
    nxt = forecast_day(red.Ws, last + pd.Timedelta(days=1), selection, c_next, cierres=red.cierres)
    return hold, nxt, cov, c_next


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    red = PL.load_network(ROOT / "data")
    horas, dias = PL.build_history(red, START, END)
    thr = PL.thresholds_from_dev(dias)
    ev = PL.evaluate_days(red, horas, dias, thr)
    dias = dias.merge(ev, on=["fecha", "linea"])
    horas.to_csv(OUT / "horas.csv.gz", index=False, float_format="%.4g", compression=GZ)
    dias.to_csv(OUT / "dias.csv.gz", index=False, float_format="%.5g", compression=GZ)
    sel = json.loads((REP / "pronostico_holdout.json").read_text(encoding="utf-8"))["seleccion_por_linea"]
    hold, nxt, cov, c_next = forecast_artifacts(red, sel)
    hold.to_csv(OUT / "pronostico_holdout.csv.gz", index=False, float_format="%.6g", compression=GZ)
    nxt.to_csv(OUT / "pronostico_siguiente.csv", index=False, float_format="%.6g")
    data_files = sorted((ROOT / "data").glob("*.xlsx")) + [ROOT / "data" / f for f in ("lineas.csv", "topologia_red.csv", "cierres_programados.csv")]
    PL.write_manifest(OUT, ROOT, data_files, {"periodo": [START, END], "umbrales": thr, "seleccion_pronostico": sel,
                                               "lineas_coherencia": list(PL.COHERENCE_LINES),
                                               "cobertura_intervalo_seleccion_2023_08_2024_06": cov,
                                               "factor_conformal": c_next,
                                               "lineas_h6_validadas": PL.H6_LINES,
                                               "dias_por_categoria": dias["categoria"].value_counts().to_dict(),
                                               "segundos_construccion": round(time.time() - t0, 1)})
    print(dias["categoria"].value_counts(), thr, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
