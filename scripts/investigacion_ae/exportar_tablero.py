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


def write(name, rows, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False), encoding="utf-8")
    print(name, len(rows))


def progreso():
    txt = (ROOT / "docs" / "investigacion_autoencoders" / "PLAN.md").read_text(encoding="utf-8")
    sec = txt.split("## Tareas", 1)[1].split("##", 1)[0]
    rows = []
    for line in sec.splitlines():
        m = re.match(r"\|\s*(AE\.\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m:
            estado = m.group(3)
            rows.append({"orden": len(rows) + 1, "id": m.group(1), "tarea": m.group(2),
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


if __name__ == "__main__":
    progreso()
    ae1()
