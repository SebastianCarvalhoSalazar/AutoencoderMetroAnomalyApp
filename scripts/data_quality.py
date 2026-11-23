"""Genera reports/calidad_datos.md (F2.3, F2.4). Uso: python scripts/data_quality.py"""
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import calendar as C
from metro_anomaly import data as D

ROOT = Path(__file__).resolve().parents[1]


def main():
    df, rep = D.load_all(ROOT / "data")
    q = D.quality_report(df)
    w = D.line_series(df, "LINEA A")
    obs = w[~w["missing_day"]]
    dt_ = pd.Series(C.day_type(obs.index), index=obs.index)
    t23 = pd.DataFrame({"tipo_dia": dt_, "nan_h23": obs["h23"].isna(), "nan_h22": obs["h22"].isna(),
                        "h23": obs["h23"], "share23": obs["h23"] / obs[D.HOUR_COLS].sum(axis=1)})
    tab23 = t23.groupby("tipo_dia").agg(dias=("nan_h23", "size"), nan_h23_pct=("nan_h23", lambda s: round(100 * s.mean(), 1)),
                                        h23_mediana=("h23", "median"), h23_pct_del_dia=("share23", lambda s: round(100 * s.mean(), 3)))
    gaps = obs[obs[D.HOUR_COLS[:-2]].isna().any(axis=1)]
    gap_rows = [f"| {d.date()} | {', '.join(c[1:] for c in D.HOUR_COLS[:-2] if np.isnan(r[c]))} | {C.holiday_name(d) or ''} |" for d, r in gaps.iterrows()]
    nan22 = obs.index[obs["h22"].isna()]
    lines = [
        "# Reporte de calidad de datos (generado por `scripts/data_quality.py`)", "",
        f"Archivos: {', '.join(rep.files)}", "",
        f"Duplicados (fecha, línea) entre archivos: {rep.cross_file_duplicates} (con valores distintos: {rep.conflicting_duplicates}); descartados: {rep.dropped}", "",
        "## Indicadores por año y línea", "", q.to_markdown(index=False), "",
        "## Línea A — días sin registro", "", ", ".join(str(d.date()) for d in w.index[w['missing_day']]) or "ninguno", "",
        "## Línea A — horas faltantes (04–21) dentro de días con registro", "",
        "| Día | Horas faltantes | Festivo |", "|---|---|---|", *gap_rows, "",
        "El `Total general` de esos días es la suma de las horas disponibles, así que subestima la afluencia real.", "",
        "## Línea A — semántica de 22:00 y 23:00 (F2.3)", "", tab23.to_markdown(), "",
        f"Días con 22:00 faltante: {', '.join(str(d.date()) for d in nan22)}", "",
    ]
    out = ROOT / "reports" / "calidad_datos.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    print(tab23)


if __name__ == "__main__":
    main()
