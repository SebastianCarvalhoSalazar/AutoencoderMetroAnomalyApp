"""AE.3 — Hipótesis A2: autoencoder multilínea frente al detector de coherencia de red (protocolo de H1).

Uso:
  python scripts/investigacion_ae/ae3_multilinea.py --linea "LINEA A"     (una línea; se corren en paralelo)
  python scripts/investigacion_ae/ae3_multilinea.py --resumen              (criterio y reportes con las 4 líneas)
Salidas: reports/investigacion_ae/runs/ae3/*.pkl, reports/investigacion_ae/ae3.json y ae3.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import data as D
from metro_anomaly.evaluation import HOLDOUT, budget_threshold, metrics, paired_recall_diff, rolling_evaluate
from metro_anomaly.red import CoherenceDetector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "investigacion_ae"
RUNS = OUT / "runs" / "ae3"
MAIN = ["LINEA 1", "LINEA 2", "LINEA A", "LINEA B", "LINEA H", "LINEA J", "LINEA K", "LINEA M", "LINEA O", "LINEA P", "LINEA T-A"]
LINES = ["LINEA A", "LINEA B", "LINEA 1", "LINEA T-A"]
KW = dict(train_start="2022-01-01", eval_start=HOLDOUT[0], eval_end=HOLDOUT[1])
KEEP = ["tasa_alertas", "tasa_alertas_festivos", "recall_moderada", "recall_severa", "recall_leve", "localizacion_hora",
        "recall_T1_moderada", "recall_T2_moderada", "recall_T3_moderada", "recall_T4_cero", "recall_T4_duplicada",
        "recall_T4_desfase", "recall_T5_perfil_festivo"]


def load_Ws():
    df, _ = D.load_all(ROOT / "data")
    return {ln: D.line_series(df, ln)[D.SCORED_HOURS] for ln in MAIN}


def run_line(ln):
    from metro_anomaly.detectors_nn import AEMultilineDetector
    RUNS.mkdir(parents=True, exist_ok=True)
    Ws = load_Ws()
    W = Ws[ln]
    t0 = time.perf_counter()
    rc = rolling_evaluate(lambda: CoherenceDetector(ln, Ws, train_start=KW["train_start"]), W, **KW)
    t1 = time.perf_counter()
    ra = rolling_evaluate(lambda: AEMultilineDetector(ln, Ws, agg="top2", seed=0), W, **KW)
    t2 = time.perf_counter()
    tag = ln.replace(" ", "_")
    for name, r in (("coherencia", rc), ("ae_multilinea", ra)):
        with open(RUNS / f"{tag}__{name}.pkl", "wb") as fh:
            pickle.dump(r, fh)
    print(ln, f"coherencia {t1 - t0:.0f}s, ae_multilinea {t2 - t1:.0f}s", flush=True)


def resumen():
    out = {"lineas": {}}
    for ln in LINES:
        tag = ln.replace(" ", "_")
        with open(RUNS / f"{tag}__coherencia.pkl", "rb") as fh:
            rc = pickle.load(fh)
        with open(RUNS / f"{tag}__ae_multilinea.pkl", "rb") as fh:
            ra = pickle.load(fh)
        tc, ta = budget_threshold(rc.real["score"]), budget_threshold(ra.real["score"])
        mc, ma = metrics(rc, tc), metrics(ra, ta)
        d = paired_recall_diff(ra, ta, rc, tc, "moderada", n_boot=2000)
        out["lineas"][ln] = {"coherencia": {k: round(float(mc.get(k, np.nan)), 4) for k in KEEP},
                             "ae_multilinea": {k: round(float(ma.get(k, np.nan)), 4) for k in KEEP},
                             "diferencia_recall_moderada": {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}}
    A = out["lineas"]["LINEA A"]
    c1 = A["diferencia_recall_moderada"]["ic_inf"] >= -0.02
    better = [ln for ln in ("LINEA B", "LINEA 1", "LINEA T-A") if out["lineas"][ln]["diferencia_recall_moderada"]["ic_inf"] > 0]
    c2 = bool(better)
    c3 = A["ae_multilinea"]["tasa_alertas_festivos"] <= A["coherencia"]["tasa_alertas_festivos"]
    out["criterio"] = {"A_no_inferior": {"cumple": bool(c1), "ic_inf": A["diferencia_recall_moderada"]["ic_inf"], "exigido": "≥ −0,02"},
                       "mejora_en_otra_linea": {"cumple": c2, "lineas": better, "exigido": "IC > 0 en B, 1 o T-A"},
                       "festivos_A": {"cumple": bool(c3), "ae": A["ae_multilinea"]["tasa_alertas_festivos"], "coherencia": A["coherencia"]["tasa_alertas_festivos"]},
                       "veredicto": "CUMPLE" if c1 and c2 and c3 else "NO CUMPLE"}
    def clean(v):
        if isinstance(v, float) and v != v:
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return v
    out = clean(out)
    (OUT / "ae3.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    md = ["# AE.3 — Hipótesis A2: autoencoder multilínea frente a coherencia de red (generado por `scripts/investigacion_ae/ae3_multilinea.py`)", "",
          "Holdout 2023-07…2024-06, inyecciones solo en la línea objetivo, mismo presupuesto (p95 de los scores reales del periodo).", ""]
    for ln, v in out["lineas"].items():
        md += [f"## {ln}", "", pd.DataFrame({"coherencia": v["coherencia"], "ae_multilinea": v["ae_multilinea"]}).to_markdown(), "",
               f"Diferencia pareada de recall moderado (ae_multilinea − coherencia): {v['diferencia_recall_moderada']}", ""]
    md += ["## Criterio pre-registrado", "", f"- Línea A no inferior (IC inf ≥ −0,02): {out['criterio']['A_no_inferior']}",
           f"- Mejora en B, 1 o T-A (IC > 0): {out['criterio']['mejora_en_otra_linea']}", f"- Festivos A no peor: {out['criterio']['festivos_A']}",
           f"- **Veredicto: {out['criterio']['veredicto']}**"]
    (OUT / "ae3.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--linea")
    ap.add_argument("--resumen", action="store_true")
    a = ap.parse_args()
    if a.resumen:
        resumen()
    else:
        run_line(a.linea)
