"""Simulación cronológica de la operación diaria multilínea (fase 3).

Para cada día t del periodo: (1) evalúa las 11 líneas del día t con `run_daily --red` (lógica de
`operacion_red.process_red`), (2) emite el pronóstico de t+1 con datos ≤ t (`forecast_red`) y (3) al final ejecuta el
monitor, que compara los pronósticos registrados con lo observado. Mide la latencia por día y escribe
reports/fase3/operacion_red.md y .json.

Uso: python scripts/fase3/simulacion_operacion_red.py --desde 2024-06-01 --hasta 2024-06-29
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import operacion_red as OR
from metro_anomaly import plataforma as PL

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "artifacts" / "plataforma" / "manifiesto.json"
OUT = ROOT / "reports" / "fase3"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2024-06-01")
    ap.add_argument("--hasta", default="2024-06-29")
    a = ap.parse_args(argv)
    red = PL.load_network(ROOT / "data")
    tmp = Path(tempfile.mkdtemp(prefix="opred_"))
    dlog, flog = tmp / "decisiones.jsonl", tmp / "pronosticos.jsonl"
    lat_eval, lat_fc, alertas = [], [], []
    for t in pd.date_range(a.desde, a.hasta):
        # Solo se conocen datos ≤ t (la red se recorta para que nada posterior pueda filtrarse).
        red_t = PL.Red({ln: W.loc[:t] for ln, W in red.Ws.items()}, red.lineas, red.topologia)
        t0 = time.perf_counter()
        s = OR.process_red([t], red_t, MAN, dlog)
        lat_eval.append(time.perf_counter() - t0)
        alertas += s["alertas"]
        t0 = time.perf_counter()
        OR.forecast_red(t + pd.Timedelta(days=1), red_t, MAN, flog)
        lat_fc.append(time.perf_counter() - t0)
        print(t.date(), f"{lat_eval[-1]:.1f}s", f"{lat_fc[-1]:.1f}s", s["alertas"], flush=True)
    mon = OR.monitor_red(dlog, flog, red)
    # Comparación con la reproducción histórica (reajuste mensual) en las mismas fechas.
    art = pd.read_csv(ROOT / "artifacts" / "plataforma" / "dias.csv.gz", parse_dates=["fecha"]).set_index(["fecha", "linea"])
    recs = [json.loads(x) for x in dlog.read_text(encoding="utf-8").splitlines()]
    same = sum(art.loc[(pd.Timestamp(r["fecha"]), r["linea"]), "categoria"] == r["categoria"] for r in recs)
    hold = pd.read_csv(ROOT / "artifacts" / "plataforma" / "pronostico_holdout.csv.gz", parse_dates=["fecha"])
    F = pd.DataFrame([json.loads(x) for x in flog.read_text(encoding="utf-8").splitlines()])
    F["fecha"] = pd.to_datetime(F["fecha"])
    M = F.merge(hold[["fecha", "linea", "hora", "observado", "pred", "p10", "p90"]], on=["fecha", "linea", "hora"], suffixes=("_dia", "_mes"))
    comp = {}
    for ln, g in M.groupby("linea"):
        y = g["observado"]
        comp[ln] = {"wape_diario": round(float((y - g["pred_dia"]).abs().sum() / y.sum()), 4),
                    "wape_mensual": round(float((y - g["pred_mes"]).abs().sum() / y.sum()), 4),
                    "cobertura_diario": round(float(((y >= g["p10_dia"]) & (y <= g["p90_dia"])).mean()), 3),
                    "cobertura_mensual": round(float(((y >= g["p10_mes"]) & (y <= g["p90_mes"])).mean()), 3) if g["p10_mes"].notna().any() else None}
    res = {"periodo": [a.desde, a.hasta], "dias": len(lat_eval), "linea_dias_registrados": len(recs),
           "coinciden_con_reproduccion_historica": int(same),
           "latencia_evaluacion_s": {"mediana": round(float(np.median(lat_eval)), 2), "p95": round(float(np.quantile(lat_eval, 0.95)), 2)},
           "latencia_pronostico_s": {"mediana": round(float(np.median(lat_fc)), 2), "p95": round(float(np.quantile(lat_fc, 0.95)), 2)},
           "alertas": alertas, "monitor": mon, "pronostico_diario_vs_mensual": comp}
    (OUT / "operacion_red.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    t = pd.DataFrame(comp).T
    md = ["# Simulación cronológica de la operación diaria multilínea (generado por `scripts/fase3/simulacion_operacion_red.py`)", "",
          f"Periodo {a.desde} … {a.hasta} ({len(lat_eval)} días). Cada día t se evalúa con datos ≤ t y se emite el pronóstico de t+1.", "",
          f"- Línea-días registrados: {len(recs)}; coinciden con la reproducción histórica (`dias.csv.gz`): **{same}/{len(recs)}**",
          f"- Latencia por día — evaluación de 11 líneas: mediana {res['latencia_evaluacion_s']['mediana']} s, p95 {res['latencia_evaluacion_s']['p95']} s; "
          f"pronóstico: mediana {res['latencia_pronostico_s']['mediana']} s, p95 {res['latencia_pronostico_s']['p95']} s",
          f"- Alertas del periodo: {', '.join(alertas) or 'ninguna'}",
          f"- Advertencias del monitor: {mon['advertencias'] or 'ninguna'}", "",
          "## Pronóstico con reajuste diario (operación) frente a reajuste mensual (reproducción del holdout)", "",
          "Mismos días y franjas; el intervalo del reajuste diario usa el factor conformal del manifiesto.", "", t.to_markdown(), ""]
    (OUT / "operacion_red.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
