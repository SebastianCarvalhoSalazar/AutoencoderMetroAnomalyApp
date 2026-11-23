"""Simulación de operación (F8): reproducción cronológica con disponibilidad temporal estricta.

Política operativa simulada (la del despliegue):
- Reentrenamiento mensual: el 1.er día de cada mes el modelo se ajusta con los datos disponibles hasta
  el día anterior (ventana de 365 días del perfil).
- Umbral fijo (congelado en DEV, el del artefacto).
- Fin de día: el día t se puntúa cuando sus 19 horas están disponibles, con contexto = días < t.
- Intra-día (supuesto NO verificado: que el dato horario llegue al cierre de cada hora): tras la hora h,
  el día t se puntúa con las horas 04..h; las demás horas se tratan como no disponibles.

La guarda de disponibilidad (``available_context``) trunca explícitamente el contexto; las funciones por
lote usan el contexto completo y se apoyan en la causalidad de los detectores (tests D8), y la simulación
verifica que ambos caminos den resultados idénticos.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SCORED_HOURS
from .detectors import REGISTRY

HOURS_INT = [int(c[1:]) for c in SCORED_HOURS]


def factory_for(detector="perfil_contextual", agg="peak_shift"):
    return lambda: REGISTRY[detector](agg=agg)


def monthly_models(W: pd.DataFrame, start, end, factory, train_start="2022-01-01") -> dict:
    """Un modelo por mes, ajustado con datos < 1.er día del mes (política de reentrenamiento mensual)."""
    out = {}
    for ms in pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), end, freq="MS"):
        hist = W.loc[train_start: ms - pd.Timedelta(days=1)]
        assert hist.index.max() < ms
        out[ms.to_period("M")] = factory().fit(hist, W.loc[: ms - pd.Timedelta(days=1)])
    return out


def available_context(W: pd.DataFrame, t: pd.Timestamp, hour: int | None = None) -> pd.DataFrame:
    """Datos disponibles al cierre de la hora ``hour`` del día t (None = cierre del día)."""
    ctx = W.loc[: t].copy()
    if hour is not None:
        ctx.loc[t, [c for c in SCORED_HOURS if int(c[1:]) > hour]] = np.nan
    return ctx


@dataclass
class EODResult:
    scores: pd.Series
    z: pd.DataFrame
    seconds_per_day: float
    guard_violations: int


def end_of_day(W: pd.DataFrame, models: dict, dates, guarded: bool = True) -> EODResult:
    """Puntuación de fin de día. Con ``guarded=True`` cada día se puntúa con el contexto truncado en t
    y se cuentan las violaciones (filas de fecha > t en el contexto, que debe ser 0)."""
    dates = pd.DatetimeIndex(dates)
    s, zs, viol = {}, [], 0
    t0 = time.perf_counter()
    for t in dates:
        det = models[t.to_period("M")]
        ctx = available_context(W, t) if guarded else W
        viol += int((ctx.index > t).sum()) if guarded else 0
        sc = det.score(W.loc[[t]], ctx)
        s[t] = float(sc.day.iloc[0])
        zs.append(sc.z)
    return EODResult(pd.Series(s), pd.concat(zs), (time.perf_counter() - t0) / max(len(dates), 1), viol)


def batch_scores(W: pd.DataFrame, models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Puntúa las filas de X (posiblemente modificadas) por lote, agrupando por mes. Devuelve score y z."""
    parts = []
    for per, idx in X.groupby(X.index.to_period("M")).groups.items():
        sc = models[per].score(X.loc[idx], W)
        parts.append(pd.concat([sc.day.rename("score"), sc.z], axis=1))
    return pd.concat(parts).sort_index()


def intraday_scores(W: pd.DataFrame, models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Score parcial tras cada hora h (filas = fechas, columnas = horas 4..22) usando solo horas ≤ h."""
    out = {}
    for h in HOURS_INT:
        Xp = X.copy()
        Xp[[c for c in SCORED_HOURS if int(c[1:]) > h]] = np.nan
        out[h] = batch_scores(W, models, Xp)["score"]
    return pd.DataFrame(out)


def first_detection(partial: pd.DataFrame, thr: float, onset_hour: pd.Series | None = None) -> pd.Series:
    """Primera hora h con score parcial > umbral (NaN si nunca). Si se da ``onset_hour``, solo cuenta h ≥ inicio."""
    res = {}
    for t, row in partial.iterrows():
        hs = [h for h in partial.columns if row[h] > thr and (onset_hour is None or h >= onset_hour.loc[t])]
        res[t] = hs[0] if hs else np.nan
    return pd.Series(res, dtype=float)
