"""Verificación visual-datos de la página multilínea: compara la tabla mostrada en el navegador (extraída por
tests/e2e/red_multilinea.js) con los artefactos `artifacts/plataforma/dias.csv.gz` y el manifiesto.

Uso: python tests/e2e/check_red_multilinea.py <dir_salida_e2e>
Sale con código 1 si hay alguna discrepancia, excepción, desbordamiento horizontal o error de JavaScript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "plataforma"


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def check(out_dir: Path) -> dict:
    res = json.loads((out_dir / "red_multilinea.json").read_text(encoding="utf-8"))
    man = json.loads((ART / "manifiesto.json").read_text(encoding="utf-8"))
    dias = pd.read_csv(ART / "dias.csv.gz", parse_dates=["fecha"])
    report = {"comparaciones": 0, "discrepancias": [], "problemas_pagina": []}
    for d, tbl in res["dias"].items():
        t = pd.Timestamp(pd.to_datetime(d, dayfirst=True))
        exp = dias[dias["fecha"] == t].set_index("linea")
        if tbl is None:
            report["discrepancias"].append(f"{d}: sin tabla")
            continue
        head = tbl["head"]
        i_res, i_sc = head.index("Resultado"), head.index("Score / umbral")
        shown = {r[0]: r for r in tbl["rows"]}
        if len(shown) != len(exp):
            report["discrepancias"].append(f"{d}: {len(shown)} filas mostradas, {len(exp)} esperadas")
        for ln, row in exp.iterrows():
            key = ln.replace("LINEA ", "")
            r = shown.get(key)
            report["comparaciones"] += 1
            if r is None:
                report["discrepancias"].append(f"{d} {ln}: falta la fila")
                continue
            label = man["categorias"][row["categoria"]]
            if label not in r[i_res]:
                report["discrepancias"].append(f"{d} {ln}: categoría mostrada '{r[i_res]}' ≠ '{label}'")
            want = f"{fmt(row['score'])} / {fmt(row['umbral'])}"
            if r[i_sc] != want:
                report["discrepancias"].append(f"{d} {ln}: score mostrado '{r[i_sc]}' ≠ '{want}'")
    for c in res["checks"]:
        if c["excepciones"] or c["desbordamiento_horizontal"]:
            report["problemas_pagina"].append(c)
    if res.get("errores_js"):
        report["problemas_pagina"].append({"errores_js": res["errores_js"]})
    marks = res.get("reproduccion_marcas", {})
    report["reproduccion_marcas"] = marks
    report["axe"] = res.get("axe")
    report["tiempos_ms"] = res.get("tiempos_ms")
    return report


if __name__ == "__main__":
    rep = check(Path(sys.argv[1]))
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    sys.exit(1 if rep["discrepancias"] or rep["problemas_pagina"] else 0)
