"""AE.1 — Reproduce desde cero el punto de partida de la línea de investigación de autoencoders.

Reentrena (rolling mensual, sin caché) `ae_contextual/top2`, `ae_contextual/max`, `ae_original/max` y el modelo vigente
`perfil_contextual/peak_shift` en DEV y en el holdout de la Línea A; fija cada umbral en el p95 de sus scores reales
de DEV; calcula métricas y diferencias pareadas; compara con `reports/holdout.md` y con la caché de la fase 1–2.

Salidas: reports/investigacion_ae/ae1.json y ae1.md; corridas en reports/investigacion_ae/runs/ (ignoradas por git).
Uso (requiere TensorFlow: requirements-research.txt): python scripts/investigacion_ae/ae1_reproducir.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly.evaluation import budget_threshold, metrics, paired_recall_diff

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "investigacion_ae"
RUNS = OUT / "runs"
CONFIGS = [("perfil_contextual", "peak_shift"), ("ae_contextual", "top2"), ("ae_contextual", "max"), ("ae_original", "max")]
REF = "perfil_contextual/peak_shift"
# Valores publicados en reports/holdout.md (fase 2). Las configuraciones con 3 semillas se publicaron como PROMEDIO de
# las semillas 0–2, así que no son comparables con una sola semilla: para ellas la prueba de reproducción es la igualdad
# exacta de scores con la corrida guardada de la semilla 0 (`max_dif_score_vs_cache`).
PUBLICADO = {"perfil_contextual/peak_shift": {"umbral": 7.388, "tasa_alertas": 0.022, "recall_moderada": 0.506, "semillas": 1},
             "ae_contextual/top2": {"umbral": 9.529, "tasa_alertas": 0.049, "recall_moderada": 0.522, "semillas": 1},
             "ae_contextual/max": {"umbral": 11.294, "tasa_alertas": 0.043, "recall_moderada": 0.485, "semillas": 3},
             "ae_original/max": {"umbral": 3578.13, "tasa_alertas": 0.024, "recall_moderada": 0.304, "semillas": 3}}


def _run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    import sys
    reusar = "--reusar" in sys.argv  # recalcula métricas con las corridas ya guardadas en RUNS (sin reentrenar)
    RUNS.mkdir(parents=True, exist_ok=True)
    res, secs = {}, {}
    prev = json.loads((OUT / "ae1.json").read_text(encoding="utf-8")).get("segundos", {}) if reusar and (OUT / "ae1.json").exists() else {}
    if not reusar:
        rev = _run_eval()
        W = rev.load_W()
    for per in ("dev", "holdout"):
        for det, agg in CONFIGS:
            if reusar:
                with open(RUNS / per / f"{det}__{agg}__s0.pkl", "rb") as fh:
                    res[(per, f"{det}/{agg}")] = pickle.load(fh)
                secs[f"{per}:{det}/{agg}"] = prev.get(f"{per}:{det}/{agg}")
                continue
            t0 = time.perf_counter()
            _, r = rev.run(per, det, agg, 0, out_dir=RUNS / per, W=W)
            res[(per, f"{det}/{agg}")] = r
            secs[f"{per}:{det}/{agg}"] = round(time.perf_counter() - t0, 1)
            print(per, det, agg, secs[f"{per}:{det}/{agg}"], "s", flush=True)
    out = {"configs": {}, "segundos": secs}
    hold_ref = res[("holdout", REF)]
    thr_ref = budget_threshold(res[("dev", REF)].real["score"])
    for det, agg in CONFIGS:
        cfg = f"{det}/{agg}"
        thr = budget_threshold(res[("dev", cfg)].real["score"])
        h = res[("holdout", cfg)]
        m = metrics(h, thr)
        keep = ["tasa_alertas", "tasa_alertas_festivos", "recall_severa", "recall_moderada", "recall_leve",
                "recall_T1_moderada", "recall_T2_moderada", "recall_T3_moderada", "recall_T4_cero", "recall_T4_duplicada",
                "recall_T4_desfase", "recall_T5_perfil_festivo", "localizacion_hora"]
        row = {"umbral_dev": round(thr, 4), **{k: round(float(m.get(k, np.nan)), 3) for k in keep}}
        if cfg != REF:
            d = paired_recall_diff(h, thr, hold_ref, thr_ref, "moderada", n_boot=2000)
            row["diferencia_recall_moderada_vs_perfil"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}
        # Reproducción frente a lo publicado y frente a la caché de la fase 2
        pub = PUBLICADO[cfg]
        row["semillas_publicadas"] = pub["semillas"]
        cache = ROOT / "reports" / "runs" / "holdout" / f"{det}__{agg}__s0.pkl"
        row["max_dif_score_vs_cache"] = None
        if cache.exists():
            with open(cache, "rb") as fh:
                old = pickle.load(fh)
            a, b = old.real["score"].to_numpy(), h.real["score"].to_numpy()
            row["max_dif_score_vs_cache"] = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else None
        same_scores = row["max_dif_score_vs_cache"] == 0.0
        same_pub = (abs(row["umbral_dev"] - pub["umbral"]) < 0.01 * max(1, pub["umbral"]) and
                    abs(row["tasa_alertas"] - pub["tasa_alertas"]) <= 0.001 and abs(row["recall_moderada"] - pub["recall_moderada"]) <= 0.001)
        # Con una semilla publicada se exige igualdad con lo publicado; con 3 (promedio), igualdad exacta con la semilla 0.
        row["reproduce_publicado"] = bool(same_pub if pub["semillas"] == 1 else same_scores)
        out["configs"][cfg] = row
        # Datos para las visualizaciones: score diario real del holdout y umbral
        out.setdefault("serie_holdout", {})[cfg] = {"fechas": [str(x.date()) for x in h.real.index],
                                                    "score": [None if np.isnan(v) else round(float(v), 3) for v in h.real["score"]],
                                                    "umbral": round(thr, 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ae1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    T = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "diferencia_recall_moderada_vs_perfil"} for k, v in out["configs"].items()}).T
    md = ["# AE.1 — Reproducción del punto de partida (generado por `scripts/investigacion_ae/ae1_reproducir.py`)", "",
          "Línea A, rolling mensual; umbral = p95 de los scores reales de DEV de cada configuración; métricas en el holdout 2023-07…2024-06.", "",
          T.to_markdown(), "", "## Diferencia pareada de recall moderado frente al perfil contextual", ""]
    for k, v in out["configs"].items():
        if "diferencia_recall_moderada_vs_perfil" in v:
            md.append(f"- {k}: {v['diferencia_recall_moderada_vs_perfil']}")
    md += ["", f"Tiempos de ejecución (s): {secs}"]
    (OUT / "ae1.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
