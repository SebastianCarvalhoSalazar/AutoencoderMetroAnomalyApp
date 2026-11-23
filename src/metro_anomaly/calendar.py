"""Calendario de Colombia y rasgos de contexto para la afluencia.

Fuente de festivos: librería ``holidays`` (versión fijada en requirements.txt), que implementa la
Ley 51 de 1983 (traslado al lunes) y los festivos religiosos móviles. La verificación contra el texto
oficial quedó bloqueada por red en este entorno (docs/BITACORA.md, F2.5); se contrasta empíricamente
con los propios datos (tests/integration/test_real_data.py::test_holiday_calendar_is_consistent_with_observed_profiles).
"""
from __future__ import annotations

from functools import lru_cache

import holidays
from dateutil.easter import easter
import numpy as np
import pandas as pd

#: Días no festivos con operación o demanda especial y predecible (víspera de Navidad y de Año Nuevo).
SPECIAL_DAYS = {(12, 24), (12, 31)}
DAY_TYPES = ("habil", "sabado", "domingo_festivo", "especial")
DAY_TYPES_EXT = DAY_TYPES + ("habil_temporada", "sabado_temporada")

#: Jornadas electorales nacionales y territoriales (fechas verificadas en registraduria.gov.co; ver
#: data/eventos_reales.csv). NO se usan como contexto del modelo: se anotan en las explicaciones.
ELECTION_DAYS = {"2019-10-27": "Elecciones de autoridades locales", "2022-03-13": "Elecciones de Congreso",
                 "2022-05-29": "Elecciones presidenciales (1.ª vuelta)", "2022-06-19": "Elecciones presidenciales (2.ª vuelta)",
                 "2023-10-29": "Elecciones territoriales"}


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


def is_holy_week(dates) -> np.ndarray:
    """Lunes a sábado de la Semana Santa (domingo de Pascua calculado con el cómputo eclesiástico)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    out = np.zeros(len(d), bool)
    for i, x in enumerate(d):
        e = pd.Timestamp(easter(x.year))
        out[i] = e - pd.Timedelta(days=6) <= x <= e - pd.Timedelta(days=1)
    return out


def is_year_end_season(dates) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.asarray(((d.month == 12) & (d.day >= 20)) | ((d.month == 1) & (d.day <= 10)))


def is_season(dates) -> np.ndarray:
    """Temporada vacacional conocida de antemano: Semana Santa o fin de año."""
    return is_holy_week(dates) | is_year_end_season(dates)


def day_type_ext(dates) -> np.ndarray:
    """Tipo de día extendido: hábil y sábado se separan en 'normal' y 'temporada'."""
    base = day_type(dates)
    season = is_season(dates)
    out = base.copy()
    out[(base == "habil") & season] = "habil_temporada"
    out[(base == "sabado") & season] = "sabado_temporada"
    return out


def event_note(date) -> str | None:
    """Anotación de calendario para explicar una alerta (festivo, elección, temporada)."""
    d = pd.Timestamp(date)
    parts = [p for p in (holiday_name(d), ELECTION_DAYS.get(str(d.date()))) if p]
    if is_holy_week([d])[0]:
        parts.append("Semana Santa")
    elif is_year_end_season([d])[0]:
        parts.append("Temporada de fin de año")
    if (d.month, d.day) in SPECIAL_DAYS:
        parts.append("Víspera de Navidad/Año Nuevo")
    return "; ".join(parts) or None


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
    f["fin_de_anio"] = is_year_end_season(d).astype(float)
    f["semana_santa"] = is_holy_week(d).astype(float)
    for k in (1, 2):
        f[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        f[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return f
