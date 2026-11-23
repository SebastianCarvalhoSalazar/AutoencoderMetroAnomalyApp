"""UX3 — Compara lo que mostró el navegador (salida de ux_audit.js) con el backend (service.py).

Uso: python tests/e2e/check_ux_audit.py <ux_audit.json> <dir_fixtures> <salida.json>
Verifica, para cada vista del histórico, las 4 métricas, la categoría y las horas de la alerta, y las filas de la
tabla de detalle; para cada carga, que el CSV descargado sea idéntico a ``summary_table`` del backend.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import service as S
from metro_anomaly.data import SCORED_HOURS

ROOT = Path(__file__).resolve().parents[2]


def num(x, d=0):  # mismo formato que app.py
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def main(audit_json, fixtures, out):
    audit = json.loads(Path(audit_json).read_text())
    hist = S.load_history(ROOT / "data")
    det, thr, doc = S.load_model(ROOT / "artifacts" / "modelo.json")
    ref = S.reference_scores(doc)
    checks = []
    steps = {s["id"]: s for s in audit["steps"]}
    for sid, s in steps.items():
        if not sid.startswith("H") or not s.get("metrics") or "date" not in s:
            continue
        r = S.score_days(det, thr, hist, [s["date"]], ref)[0]
        exp_metrics = [num(r.score, 2), num(thr, 2), "—" if r.exceedance is None else f"{num(100 * r.exceedance, 1)} %",
                       f"{'+' if r.deviation_pct >= 0 else ''}{num(r.deviation_pct, 1)} %"]
        alert = " ".join(s["alerts"])
        ok_cat = r.status in alert if r.category != "posible_problema_de_datos" else "Posible problema de datos" in alert
        # En "posible problema de datos" la alerta lista las banderas de calidad (no las horas): no aplica.
        ok_hours = all(f"{c[1:]} h" in alert for c in r.hour_flags) if r.category == "anomalia_estadistica" else None
        table = "\n".join(s.get("tableText", []))
        # La tabla está dentro de un expansor: solo es verificable cuando se abrió (paso H14).
        rows_ok = (all(num(r.observed[c]) in table and num(r.expected[c]) in table and num(r.z[c], 2) in table
                       for c in SCORED_HOURS) if table.strip() else None)
        checks.append({"vista": sid, "metricas_ui": s["metrics"][:4], "metricas_backend": exp_metrics,
                       "metricas_ok": s["metrics"][:4] == exp_metrics, "categoria_ok": ok_cat, "horas_ok": ok_hours,
                       "tabla_ok": rows_ok})
    for sid, s in steps.items():
        if not s.get("download"):
            continue
        f = Path(fixtures) / s["file"]
        new = S.parse_upload(f.read_bytes(), f.name)
        ctx = S.merge_context(hist, new)
        res = []
        for d in [d for d in new.index if new.loc[d, SCORED_HOURS].notna().any()]:
            try:
                res.extend(S.score_days(det, thr, ctx, [d], ref))
            except S.InputError:
                pass
        backend = S.summary_table(res).to_csv(index=False)
        ui = Path(s["download"]).read_text(encoding="utf-8")
        checks.append({"vista": sid, "csv_identico": ui == backend, "filas": len(res),
                       "diferencias": None if ui == backend else
                       str(pd.read_csv(io.StringIO(ui)).compare(pd.read_csv(io.StringIO(backend))).head())})
    ok = all(all(v for k, v in c.items() if (k.endswith("_ok") or k == "csv_identico") and v is not None) for c in checks)
    Path(out).write_text(json.dumps({"todo_coincide": ok, "verificaciones": checks}, indent=1, ensure_ascii=False))
    for c in checks:
        print({k: v for k, v in c.items() if k not in ("metricas_ui", "metricas_backend", "diferencias")})
    print("TODO COINCIDE" if ok else "HAY DIFERENCIAS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
