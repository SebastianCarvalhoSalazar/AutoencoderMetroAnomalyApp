"""F4.2 — Selección de modelo en DEV con la regla pre-registrada (docs/PLAN_MAESTRO.md, F4.2).

Regla:
1. Umbral de cada configuración = cuantil 95 % de sus scores de días reales de DEV (presupuesto del 5 %).
2. Restricción: tasa de alertas en festivos de DEV ≤ 10 %.
3. Criterio primario: recall medio de los escenarios moderados (T1, T2, T3).
4. Empate (diferencia < 0,02 con IC 95 % solapados) → la configuración más simple.
Salidas: reports/seleccion_dev.md, configs/seleccion.yaml (modelo + umbral congelados).
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metro_anomaly.evaluation import block_bootstrap, budget_threshold, metrics

ROOT = Path(__file__).resolve().parents[1]
SIMPLICITY = ["naive_semanal", "perfil_robusto", "perfil_contextual", "regresion_calendario", "regresion_contextual",
              "isolation_forest", "ae_original", "ae_contextual"]
AGG_SIMPLICITY = {"max": 0, "top2": 1, "peak_shift": 2}
KEYS = ["tasa_alertas", "tasa_alertas_festivos", "recall_moderada_media", "recall_severa", "recall_moderada", "recall_leve",
        "recall_T1_moderada", "recall_T2_moderada", "recall_T3_moderada", "recall_T4_cero", "recall_T4_duplicada",
        "recall_T4_desfase", "recall_T5_perfil_festivo", "localizacion_hora"]


def load_runs(period="dev"):
    runs = defaultdict(list)
    for f in sorted((ROOT / "reports" / "runs" / period).glob("*.pkl")):
        det, agg, seed = f.stem.split("__")
        with open(f, "rb") as fh:
            runs[(det, agg)].append((int(seed[1:]), pickle.load(fh)))
    return runs


def evaluate_config(seed_runs):
    rows, thrs = [], []
    for seed, res in sorted(seed_runs):
        thr = budget_threshold(res.real["score"])
        m = metrics(res, thr)
        m["seed"], m["umbral"] = seed, thr
        rows.append(m)
        thrs.append(thr)
    df = pd.DataFrame(rows)
    seed0 = sorted(seed_runs)[0][1]
    ci = block_bootstrap(seed0, thrs[0], ["recall_moderada_media", "tasa_alertas_festivos"], n_boot=1000)
    return df, ci


def main():
    runs = load_runs("dev")
    table, details = [], {}
    for (det, agg), sr in runs.items():
        df, ci = evaluate_config(sr)
        mean = df[KEYS].mean()
        row = {"detector": det, "agg": agg, "n_semillas": len(df),
               **{k: round(float(mean[k]), 3) for k in KEYS},
               "recall_mod_ic_inf": round(float(ci.loc["recall_moderada_media", "ic_inf"]), 3),
               "recall_mod_ic_sup": round(float(ci.loc["recall_moderada_media", "ic_sup"]), 3),
               "recall_mod_sd_semillas": round(float(df["recall_moderada_media"].std(ddof=0)), 3),
               "umbral_semilla0": float(df["umbral"].iloc[0])}
        table.append(row)
        details[(det, agg)] = df
    T = pd.DataFrame(table)
    T["simplicidad"] = T["detector"].map(SIMPLICITY.index) * 3 + T["agg"].map(AGG_SIMPLICITY)
    T = T.sort_values("recall_moderada_media", ascending=False).reset_index(drop=True)
    elig = T[T["tasa_alertas_festivos"] <= 0.10]
    if elig.empty:
        raise SystemExit("Ninguna configuración cumple la restricción de festivos; ver tabla")
    best = elig.iloc[0]
    tie = elig[(elig["recall_moderada_media"] >= best["recall_moderada_media"] - 0.02)
               & (elig["recall_mod_ic_sup"] >= best["recall_mod_ic_inf"])]
    chosen = tie.sort_values("simplicidad").iloc[0]
    decision = {
        "detector": str(chosen["detector"]), "agg": str(chosen["agg"]),
        "umbral": float(chosen["umbral_semilla0"]),
        "presupuesto_alertas": 0.05,
        "procedencia_umbral": "cuantil 0,95 de los scores fuera de muestra de los días reales de DEV (2022-04-01…2023-06-30), rolling mensual",
        "mejor_por_recall": f"{best['detector']}/{best['agg']} ({best['recall_moderada_media']})",
        "empatados": [f"{r.detector}/{r.agg}" for r in tie.itertuples()],
        "baselines_umbral": {f"{r.detector}/{r.agg}": float(r.umbral_semilla0) for r in T.itertuples()},
    }
    (ROOT / "configs").mkdir(exist_ok=True)
    with open(ROOT / "configs" / "seleccion.yaml", "w") as fh:
        yaml.safe_dump(decision, fh, allow_unicode=True, sort_keys=False)
    cols = ["detector", "agg", "n_semillas", "recall_moderada_media", "recall_mod_ic_inf", "recall_mod_ic_sup",
            "recall_mod_sd_semillas", "tasa_alertas_festivos", "recall_severa", "recall_leve", "recall_T1_moderada",
            "recall_T2_moderada", "recall_T3_moderada", "recall_T4_cero", "recall_T4_duplicada", "recall_T4_desfase",
            "recall_T5_perfil_festivo", "localizacion_hora"]
    md = ["# Selección de modelo en DEV (generado por `scripts/select_model.py`)", "",
          "Periodo evaluado: 2022-04-01…2023-06-30 (rolling mensual, ajuste desde 2022-01-01). Umbral por configuración: "
          "cuantil 95 % de sus scores de días reales (tasa de alertas = 5 % por construcción). IC 95 %: bootstrap por semanas (semilla 0).", "",
          T[cols].to_markdown(index=False), "",
          "## Decisión", "", "```yaml", yaml.safe_dump(decision, allow_unicode=True, sort_keys=False), "```"]
    (ROOT / "reports" / "seleccion_dev.md").write_text("\n".join(md), encoding="utf-8")
    print(T[cols].to_string())
    print(decision)


if __name__ == "__main__":
    main()
