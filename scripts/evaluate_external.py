"""F4.4 — Evaluación externa (D6, D7) con el modelo y los umbrales congelados en DEV.

- ext2019 (transferencia): ajuste con DEV, nivel causal con datos de 2019; evaluación 2019-03…2019-12
  (≥ 8 semanas de calentamiento para el nivel). D6: tasa de alertas ≤ 15 %.
- covid (causal): ajuste desde 2019-01-01, rolling mensual 2020–2021. Comportamiento ante EV1/EV2.
- D7: eventos prospectivos con fuente oficial (data/eventos_reales.csv, EV1–EV4): detección en la fecha
  de inicio o en los 2 días siguientes. Meta: ≥ 2/3 de los eventos verificados.
Salida: reports/externo.md
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metro_anomaly.calendar import event_note
from metro_anomaly.evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]
PROSPECTIVE = ["EV1", "EV2", "EV3", "EV4"]


def load(period, cfg, seed=0):
    det, agg = cfg.split("/")
    f = ROOT / "reports" / "runs" / period / f"{det}__{agg}__s{seed}.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def event_detection(res, thr, start):
    """(detectado, primer día con alerta, score máximo) en [inicio, inicio+2]."""
    s = res.real["score"]
    win = s.loc[pd.Timestamp(start): pd.Timestamp(start) + pd.Timedelta(days=2)]
    if win.empty:
        return None, None, np.nan
    hit = win[win > thr]
    return bool(len(hit)), (hit.index[0].date() if len(hit) else None), float(win.max())


def main():
    sel = yaml.safe_load(open(ROOT / "configs" / "seleccion.yaml"))
    chosen = f"{sel['detector']}/{sel['agg']}"
    thrs = sel["baselines_umbral"]
    cfgs = [chosen, "perfil_robusto/max", "regresion_contextual/max", "naive_semanal/max", "ae_original/max"]
    ev = pd.read_csv(ROOT / "data" / "eventos_reales.csv")
    md = ["# Evaluación externa (generado por `scripts/evaluate_external.py`)", "",
          f"Configuración elegida: **{chosen}** (umbral DEV {thrs[chosen]:.4f}). Cada configuración usa su propio umbral de DEV.", ""]

    rows = []
    for cfg in cfgs:
        r = load("ext2019", cfg)
        if r is None:
            continue
        m = metrics(r, thrs[cfg])
        rows.append({"config": cfg, "tasa_alertas": m["tasa_alertas"], "tasa_festivos": m["tasa_alertas_festivos"],
                     "recall_severa": m["recall_severa"], "recall_moderada": m["recall_moderada"],
                     "recall_T5": m["recall_T5_perfil_festivo"], "D6_cumple(≤15%)": m["tasa_alertas"] <= 0.15})
    md += ["## 2019 — transferencia (D6)", "", "Ajuste con DEV (2022-01…2023-06); nivel y referencia causales sobre 2019; evaluación 2019-03-01…2019-12-31 "
           "(días con las 19 horas completas). Advertencia: no es un backtest causal en el calendario (el modelo de forma se aprende con años posteriores).", "",
           pd.DataFrame(rows).round(3).to_markdown(index=False), ""]

    r19 = load("ext2019", chosen)
    al = r19.real[r19.real["score"] > thrs[chosen]]
    md += [f"Alertas del modelo elegido en 2019 ({len(al)}):", "",
           pd.DataFrame({"día": al.index.day_name(), "score": al["score"].round(2),
                         "calendario": [event_note(d) or "" for d in al.index]}, index=al.index.date).to_markdown(), ""]

    rows = []
    for cfg in cfgs:
        r = load("covid", cfg)
        if r is None:
            continue
        a = (r.real["score"] > thrs[cfg]).groupby(r.real["mes"]).mean()
        rows.append(pd.Series(a, name=cfg))
    covid = pd.DataFrame(rows).T
    md += ["## 2020–2021 — estrés COVID (causal, rolling mensual desde 2019)", "", "Tasa de alertas mensual:", "",
           covid.round(2).to_markdown(), ""]

    d7 = []
    for _, e in ev[ev["id"].isin(PROSPECTIVE)].iterrows():
        period = {"EV1": "covid", "EV2": "covid", "EV3": "ext2019", "EV4": "holdout"}[e["id"]]
        row = {"evento": e["id"], "fecha": e["fecha_inicio"], "periodo": period, "descripción": e["descripcion"][:70]}
        for cfg in cfgs:
            r = load(period, cfg)
            if r is None:
                row[cfg] = "n/d"
                continue
            hit, first, smax = event_detection(r, thrs[cfg], e["fecha_inicio"])
            row[cfg] = "sin datos" if hit is None else (f"✔ {first} (max {smax:.1f})" if hit else f"✘ (max {smax:.1f})")
        d7.append(row)
    D7 = pd.DataFrame(d7)
    n_ok = sum(str(v).startswith("✔") for v in D7[chosen])
    n_eval = sum(str(v)[:1] in "✔✘" for v in D7[chosen])
    md += ["## D7 — eventos reales prospectivos con fuente oficial", "",
           "Detección = score > umbral en la fecha de inicio o en los 2 días siguientes.", "", D7.to_markdown(index=False), "",
           f"**Modelo elegido: {n_ok}/{n_eval} eventos detectados** (meta ≥ 2/3 → {'✔' if n_eval >= 3 and n_ok / n_eval >= 2 / 3 else '✘'}).", ""]

    s = load("covid", chosen).real["score"]
    md += ["Scores diarios del modelo elegido alrededor de EV1/EV2 (marzo y abril de 2020):", "",
           s.loc["2020-03-10":"2020-04-15"].round(2).to_frame().assign(alerta=lambda d: d["score"] > thrs[chosen]).to_markdown(), ""]
    (ROOT / "reports" / "externo.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
