"""Fuente climática: Open-Meteo (reanálisis), alineada a las franjas horarias de afluencia.

Alineación (documentación de Open-Meteo: precipitación = suma de la hora precedente): la lluvia de la franja
h (h:00–h:59) es el registro con marca h+1. Ver data/clima_open_meteo_medellin.meta.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import HOURS

COLUMNS = {"precipitacion_mm_hora_previa": "lluvia_mm", "temperatura_c": "temp_c"}


def load_hourly(path: str | Path) -> pd.DataFrame:
    """Serie horaria local (índice = inicio de la franja horaria), con lluvia de esa franja en mm."""
    df = pd.read_csv(path, parse_dates=["tiempo_local"]).set_index("tiempo_local").sort_index()
    rain = df["precipitacion_mm_hora_previa"].shift(-1)  # valor en h+1 = lluvia durante [h, h+1)
    temp = df["temperatura_c"]
    out = pd.DataFrame({"lluvia_mm": rain, "temp_c": temp})
    return out


def daily_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Rasgos diarios en las franjas de operación (04–22): lluvia total, lluvia en las horas pico, máxima horaria."""
    h = hourly.copy()
    h["fecha"] = h.index.normalize()
    h["hora"] = h.index.hour
    op = h[h["hora"].isin(HOURS[:-1])]
    pico = op[op["hora"].isin([6, 7, 17, 18])]
    return pd.DataFrame({
        "lluvia_dia_mm": op.groupby("fecha")["lluvia_mm"].sum(),
        "lluvia_pico_mm": pico.groupby("fecha")["lluvia_mm"].sum(),
        "lluvia_max_hora_mm": op.groupby("fecha")["lluvia_mm"].max(),
        "temp_media_c": op.groupby("fecha")["temp_c"].mean(),
    })


def hourly_matrix(hourly: pd.DataFrame, dates) -> pd.DataFrame:
    """Matriz fechas × franjas (h04…h22) de lluvia en mm, alineada a la afluencia."""
    h = hourly["lluvia_mm"]
    rows = {}
    for d in pd.DatetimeIndex(dates):
        rows[d] = [h.get(d + pd.Timedelta(hours=k), np.nan) for k in HOURS[:-1]]
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"h{k:02d}" for k in HOURS[:-1]])
