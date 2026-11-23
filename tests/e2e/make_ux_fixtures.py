"""Genera los archivos de prueba de UX/E2E (formato oficial) en el directorio indicado.
Uso: python tests/e2e/make_ux_fixtures.py <dir_salida>"""
import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import data as D

ROOT = Path(__file__).resolve().parents[2]


def official(rows: pd.DataFrame, line="LÍNEA A", fmt="xlsx", path=None, time_cols=True):
    df = pd.DataFrame({"Día": rows.index.strftime("%d/%m/%Y"), "Línea de Servicio": line})
    for h, c in zip(D.HOURS, D.HOUR_COLS):
        df[dt.time(h) if time_cols else f"{h:02d}:00:00"] = rows[c].to_numpy()
    df[D.TOTAL_COL] = rows[D.HOUR_COLS].sum(axis=1, min_count=1).to_numpy()
    if fmt == "xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df, _ = D.load_all(ROOT / "data")
    W = D.line_series(df, "LINEA A")[D.HOUR_COLS]
    base = W.loc["2024-05-01":"2024-06-30"].copy()
    official(base, path=out / "01_normal_may_jun_2024.xlsx")
    official(base, fmt="csv", time_cols=False, path=out / "02_normal_may_jun_2024.csv")
    a = base.copy(); a.loc["2024-06-12", ["h17", "h18"]] *= 0.1
    official(a, path=out / "03_anomalia_inyectada_2024-06-12.xlsx")
    m = base.copy(); m.loc["2024-06-12", ["h10", "h11"]] = np.nan
    official(m, path=out / "04_horas_faltantes_2024-06-12.xlsx")
    fut = W.loc["2023-07-03":"2023-09-30"].copy(); fut.index = fut.index + pd.Timedelta(days=364 * 1 + 0)
    fut = fut.loc[fut.index > W.index.max()]
    fut.loc[fut.index[-10], ["h06", "h07", "h08"]] *= 0.3
    official(fut, path=out / "05_futuro_con_contexto.xlsx")
    far = W.loc["2023-07-03":"2023-07-05"].copy(); far.index = far.index + pd.Timedelta(days=728)
    official(far, path=out / "06_historia_insuficiente.xlsx")
    official(base.iloc[:5], line="LÍNEA B", path=out / "07_otra_linea.xlsx")
    n = base.iloc[:5].copy(); n.iloc[2, 5] = -10
    official(n, path=out / "08_valores_negativos.xlsx")
    (out / "09_esquema_invalido.csv").write_text("fecha,valor\n2024-06-01,10\n", encoding="utf-8")
    (out / "10_corrupto.xlsx").write_bytes(np.random.default_rng(0).bytes(4096))
    (out / "11_vacio.csv").write_bytes(b"")
    e = base.copy(); e.loc["2024-06-12", D.HOUR_COLS] = e.loc["2024-06-12", D.HOUR_COLS] * 1000
    official(e, path=out / "12_valores_extremos.xlsx")
    z = base.copy(); z.loc["2024-06-12", D.HOUR_COLS] = 0
    official(z, path=out / "13_dia_en_ceros.xlsx")
    t = pd.read_excel(out / "01_normal_may_jun_2024.xlsx")
    c9 = next(c for c in t.columns if str(c).startswith("09:00"))
    t[c9] = t[c9].astype(object); t.loc[3, c9] = "N/D"
    t.to_excel(out / "14_texto_en_numeros.xlsx", index=False)
    dup = pd.read_excel(out / "01_normal_may_jun_2024.xlsx"); pd.concat([dup, dup.iloc[[0]]]).to_excel(out / "15_filas_duplicadas.xlsx", index=False)
    shutil.copy(ROOT / "data" / "Afluencia_Metro_2024.xlsx", out / "16_archivo_oficial_2024_todas_las_lineas.xlsx")
    (out / "17_extension_no_soportada.txt").write_text("hola")
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1])
