"""Exporta los resultados de la investigación de autoencoders a archivos JSON `{rows, meta}` para el tablero
de visualizaciones. El avance se lee de docs/investigacion_autoencoders/PLAN.md (única fuente).

Salida: reports/investigacion_ae/tablero/*.json
Uso: python scripts/investigacion_ae/exportar_tablero.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IA = ROOT / "reports" / "investigacion_ae"
OUT = IA / "tablero"
LABEL = {"perfil_contextual/peak_shift": "Perfil contextual (vigente)", "ae_contextual/top2": "AE con calendario (top2)",
         "ae_contextual/max": "AE con calendario (max)", "ae_original/max": "AE original"}


def code_meta(*scripts):
    return [{"label": s, "value": (ROOT / s).read_text(encoding="utf-8"), "type": "code", "lang": "python"} for s in scripts]


def _clean(v):
    """JSON estricto: NaN e infinitos no son JSON válido (el navegador rechaza el archivo); se escriben como null."""
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def write(name, rows, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    txt = json.dumps({"rows": _clean(rows), "meta": meta}, ensure_ascii=False, allow_nan=False)
    (OUT / f"{name}.json").write_text(txt, encoding="utf-8")
    print(name, len(rows))


def progreso():
    txt = (ROOT / "docs" / "investigacion_autoencoders" / "PLAN.md").read_text(encoding="utf-8")
    sec = txt.split("## Tareas", 1)[1].split("##", 1)[0]
    rows = []
    for line in sec.splitlines():
        m = re.match(r"\|\s*(AE\.\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m:
            estado = m.group(3)
            rows.append({"orden": len(rows) + 1, "id": m.group(1), "tarea": m.group(2).replace("`", ""),
                         "estado": estado.split(" ")[0].strip("*"), "detalle": estado})
    write("progreso", rows, [{"label": "Fuente", "value": "docs/investigacion_autoencoders/PLAN.md, sección Tareas"}])


def ae1():
    p = IA / "ae1.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    meta = code_meta("scripts/investigacion_ae/ae1_reproducir.py") + [
        {"label": "Periodo", "value": "Línea A; umbral = p95 de los scores reales de DEV (2022-04…2023-06); métricas en el holdout 2023-07…2024-06"}]
    rows, diffs, serie = [], [], []
    for orden, (cfg, v) in enumerate(d["configs"].items(), 1):
        rows.append({"orden": orden, "config": cfg, "modelo": LABEL.get(cfg, cfg),
                     **{k: v[k] for k in v if not isinstance(v[k], dict)}})
        if "diferencia_recall_moderada_vs_perfil" in v:
            x = v["diferencia_recall_moderada_vs_perfil"]
            diffs.append({"orden": orden, "config": cfg, "modelo": LABEL.get(cfg, cfg), "diferencia": x["diferencia"],
                          "ic_inf": x["ic_inf"], "ic_sup": x["ic_sup"], "n": x["n"]})
    for cfg, s in d["serie_holdout"].items():
        thr = s["umbral"]
        for f, sc in zip(s["fechas"], s["score"]):
            if sc is not None:
                serie.append({"config": cfg, "modelo": LABEL.get(cfg, cfg), "fecha": f, "score": sc, "umbral": thr,
                              "relativo": round(sc / thr, 4), "alerta": sc > thr})
    write("ae1_metricas", rows, meta)
    write("ae1_diferencias", diffs, meta)
    write("ae1_serie", serie, meta)


def explicar():
    p = IA / "explicar.json"
    if not p.exists():
        return
    import numpy as np
    d = json.loads(p.read_text(encoding="utf-8"))
    meta = code_meta("scripts/investigacion_ae/ae_explicar.py") + [
        {"label": "Modelos", "value": "Los mismos de la evaluación rolling: ajuste con datos anteriores al mes del día mostrado, semilla 0; "
                                      "umbrales fijados en 2022-04…2023-06"}]
    write("ex_ejemplos", d["ejemplos"], meta)
    write("ex_horas", d["ejemplos_horas"], meta)
    lm = d["latente_meta"]
    write("ex_latente", d["latente"], meta + [{"label": "Proyección", "value": f"PCA de las {lm['dim_cuello']} coordenadas del cuello de botella; "
                                                                               f"varianza explicada {lm['varianza_pc'][0]:.1%} y {lm['varianza_pc'][1]:.1%}"}])
    # Distribución: resumen por cuantiles (el detalle está en explicar.json)
    rows = []
    groups = ["Días reales 2022–2023 (fijan el umbral)", "Días reales del periodo evaluado", "Anomalías simuladas leves",
              "Anomalías simuladas moderadas", "Anomalías simuladas severas"]
    for cfg in ("ae_contextual/top2", "perfil_contextual/peak_shift"):
        for gi, g in enumerate(groups):
            v = np.array([r["rel"] for r in d["distribucion"] if r["config"] == cfg and r["grupo"] == g])
            if len(v):
                q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
                rows.append({"config": cfg, "modelo": LABEL[cfg], "orden": gi + 1, "grupo": g, "n": int(len(v)),
                             "sobre_umbral": round(float((v > 1).mean()), 4), **{f"p{k}": round(float(x), 4) for k, x in zip((5, 25, 50, 75, 95), q)}})
    write("ex_distribucion", rows, meta + [{"label": "Cálculo", "value": "Cuantiles 5, 25, 50, 75 y 95 de score/umbral por grupo; "
                                                                          "sobre_umbral = proporción de días con score mayor que el umbral"}])
    write("ex_mapa", d["mapa"], meta)


if __name__ == "__main__":
    progreso()
    ae1()
    explicar()
