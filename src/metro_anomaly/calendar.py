"""Calendario de Colombia y rasgos de contexto para la afluencia.

Fuente de festivos: librería ``holidays`` (versión fijada en requirements.txt), que implementa la
Ley 51 de 1983 (traslado al lunes) y los festivos religiosos móviles. La verificación contra el texto
oficial quedó bloqueada por red en este entorno (docs/BITACORA.md, F2.5); se contrasta empíricamente
con los propios datos (tests/integration/test_calendar_data.py).
"""
from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

#: Días no festivos con operación o demanda especial y predecible (víspera de Navidad y de Año Nuevo).
SPECIAL_DAYS = {(12, 24), (12, 31)}
DAY_TYPES = ("habil", "sabado", "domingo_festivo", "especial")


@lru_cache(maxsize=None)
def _co(years: tuple[int, ...]):
    return holidays.Colombia(years=list(years))


def is_holiday(dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    co = _co(tuple(range(int(d.year.min()), int(d.year.max()) + 1))) if len(d) else {}
    return np.array([x.date() in co for x in d], dtype=bool)


def holiday_name(date) -> str | None:
    d = pd.Timestamp(date)
    return _co((d.year,)).get(d.date())


def day_type(dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    fest = is_holiday(d)
    special = np.array([(x.month, x.day) in SPECIAL_DAYS for x in d], dtype=bool)
    out = np.where(d.dayofweek < 5, "habil", np.where(d.dayofweek == 5, "sabado", "domingo_festivo")).astype(object)
    out[fest] = "domingo_festivo"
    out[special & ~fest] = "especial"
    return out


def calendar_features(dates) -> pd.DataFrame:
    """Rasgos de calendario usados por los detectores (todos conocidos de antemano: no hay fuga)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    fest = is_holiday(d)
    doy = d.dayofyear.to_numpy()
    f = pd.DataFrame(index=d)
    for k in range(7):
        f[f"dow_{k}"] = (d.dayofweek == k).astype(float)
    f["festivo"] = fest.astype(float)
    dt_ = day_type(d)
    f["especial"] = (dt_ == "especial").astype(float)
    nxt, prv = is_holiday(d + pd.Timedelta(days=1)), is_holiday(d - pd.Timedelta(days=1))
    f["vispera_festivo"] = (nxt & ~fest).astype(float)
    f["post_festivo"] = (prv & ~fest).astype(float)
    f["fin_de_anio"] = (((d.month == 12) & (d.day >= 20)) | ((d.month == 1) & (d.day <= 10))).astype(float)
    for k in (1, 2):
        f[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        f[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return f
