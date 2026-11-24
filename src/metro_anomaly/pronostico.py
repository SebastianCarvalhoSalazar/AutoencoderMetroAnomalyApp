"""Pronóstico de entradas por línea y franja horaria para el día siguiente (F10, hipótesis H5/H5b/H6).

Diseño causal: el pronóstico del día t usa solo información disponible al cierre del día t-1 (nivel reciente,
último día comparable) y el calendario de t (conocido de antemano). El clima observado del día t solo se usa
como variante "oráculo" (cota superior del aporte del clima; en operación requeriría un pronóstico del tiempo).

Modelos comparados:
- ``naive``: el valor de la misma franja en el último día anterior del mismo tipo (sin modelo).
- ``perfil``: perfil contextual por línea (mediana por tipo de día × hora + nivel causal).
- ``gbm``: gradient boosting global (todas las líneas) sobre log1p − nivel, con calendario, hora, línea y rezagos;
  intervalos por regresión cuantílica (p10, p90).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .calendar import DAY_TYPES_EXT, calendar_features, day_type_ext
from .data import SCORED_HOURS
from .detectors import NaiveSeasonalDetector, ProfileContextDetector, level_series

HOURS_INT = [int(c[1:]) for c in SCORED_HOURS]


def _lag_same_type(W: pd.DataFrame, dates) -> pd.DataFrame:
    """log1p del último día < t del mismo tipo de día básico con datos completos (hasta 35 días atrás)."""
    return NaiveSeasonalDetector._reference(dates, W)


def _lag_same_type_ext(W: pd.DataFrame, dates) -> pd.DataFrame:
    """Como ``_lag_same_type`` pero con el tipo de día extendido: un día hábil normal no toma como referencia
    un día de Semana Santa o de fin de año (y viceversa). Si no hay candidato en 35 días, usa el tipo básico."""
    ctx = W[SCORED_HOURS].dropna(how="any")
    types = pd.Series(day_type_ext(ctx.index), index=ctx.index)
    basic = _lag_same_type(W, dates)
    out = basic.copy()
    for t, tt in zip(pd.DatetimeIndex(dates), day_type_ext(dates)):
        prev = types.loc[t - pd.Timedelta(days=35): t - pd.Timedelta(days=1)]
        cand = prev.index[prev == tt]
        if len(cand):
            out.loc[t] = np.log1p(ctx.loc[cand[-1]].to_numpy())
    return out


def build_rows(Ws: dict, dates, weather_hourly: pd.DataFrame | None = None, include_future: bool = False) -> pd.DataFrame:
    """Filas largas (fecha, línea, hora) con objetivo y rasgos causales.

    ``include_future``: incluye también fechas posteriores al último dato de cada línea (objetivo NaN), para
    emitir pronósticos; los rasgos solo usan datos < fecha."""
    dates = pd.DatetimeIndex(dates)
    cal = calendar_features(dates)
    dte = pd.Series(day_type_ext(dates), index=dates)
    parts = []
    for li, (ln, W) in enumerate(sorted(Ws.items())):
        W = W[SCORED_HOURS]
        d_ok = dates[dates.isin(W.index) | (include_future & (dates > W.index.max()))]
        if len(d_ok) == 0:
            continue
        lvl = level_series(W, d_ok, exclude_season=True)
        lag = _lag_same_type(W, d_ok)
        lag_ext = _lag_same_type_ext(W, d_ok)
        y = np.log1p(W.reindex(d_ok))
        for hi, h in enumerate(SCORED_HOURS):
            df = pd.DataFrame(index=d_ok)
            df["fecha"] = d_ok
            df["linea"] = ln
            df["linea_id"] = li
            df["hora"] = HOURS_INT[hi]
            df["nivel"] = lvl.values
            df["lag_mismo_tipo"] = lag[h].values - lvl.values
            df["lag_mismo_tipo_ext"] = lag_ext[h].values - lvl.values
            df["y_log"] = y[h].values
            df["y_rel"] = df["y_log"] - df["nivel"]
            parts.append(df)
    R = pd.concat(parts, ignore_index=True)
    C = cal.reindex(R["fecha"]).reset_index(drop=True)
    R = pd.concat([R, C], axis=1)
    for t in DAY_TYPES_EXT:
        R[f"tipo_{t}"] = (dte.reindex(R["fecha"]).values == t).astype(float)
    if weather_hourly is not None:
        rain = weather_hourly["lluvia_mm"]
        R["lluvia_franja"] = [rain.get(f + pd.Timedelta(hours=int(h)), np.nan) for f, h in zip(R["fecha"], R["hora"])]
        daily = weather_hourly["lluvia_mm"].groupby(weather_hourly.index.normalize()).sum()
        R["lluvia_dia"] = daily.reindex(R["fecha"]).values
    return R


FEATURES_BASE = ["linea_id", "hora", "lag_mismo_tipo"] + [f"dow_{k}" for k in range(7)] + [
    "festivo", "especial", "vispera_festivo", "post_festivo", "fin_de_anio", "semana_santa",
    "sin_doy_1", "cos_doy_1", "sin_doy_2", "cos_doy_2"] + [f"tipo_{t}" for t in DAY_TYPES_EXT]
FEATURES_CLIMA = ["lluvia_franja", "lluvia_dia"]
#: Variante v2 (definida tras observar el error de abril de 2024; seleccionada solo con DEV): rezago por tipo extendido.
FEATURES_V2 = [f if f != "lag_mismo_tipo" else "lag_mismo_tipo_ext" for f in FEATURES_BASE]


class GBMForecaster:
    def __init__(self, features, seed=0, max_iter=300):
        self.features, self.seed, self.max_iter = features, seed, max_iter

    def fit(self, R: pd.DataFrame):
        tr = R.dropna(subset=["y_rel", "nivel"])
        X, y = tr[self.features].to_numpy(float), tr["y_rel"].to_numpy(float)
        kw = dict(max_iter=self.max_iter, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=30,
                  categorical_features=[self.features.index("linea_id")], random_state=self.seed)
        self.m_ = HistGradientBoostingRegressor(loss="absolute_error", **kw).fit(X, y)
        self.q10_ = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **kw).fit(X, y)
        self.q90_ = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **kw).fit(X, y)
        return self

    def predict(self, R: pd.DataFrame) -> pd.DataFrame:
        X = R[self.features].to_numpy(float)
        base = R["nivel"].to_numpy(float)
        out = pd.DataFrame({"pred": np.expm1(self.m_.predict(X) + base), "p10": np.expm1(self.q10_.predict(X) + base),
                            "p90": np.expm1(self.q90_.predict(X) + base)}, index=R.index)
        out["p10"] = np.minimum(out["p10"], out["pred"])
        out["p90"] = np.maximum(out["p90"], out["pred"])
        return out


def profile_predict(Ws: dict, hist_end, dates) -> dict:
    """Perfil contextual por línea (el modelo actual), ajustado con datos ≤ hist_end."""
    out = {}
    for ln, W in Ws.items():
        W = W[SCORED_HOURS]
        hist = W.loc["2022-01-01": hist_end]
        dd = pd.DatetimeIndex(dates)
        d_ok = dd[dd.isin(W.index) | (dd > W.index.max())]  # fechas con dato o posteriores al último dato (pronóstico)
        try:
            det = ProfileContextDetector(agg="max").fit(hist, W.loc[:hist_end])
        except ValueError:
            continue
        out[ln] = np.expm1(det.expected_log(d_ok, W))
    return out


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    return float(np.abs(y[m] - yhat[m]).sum() / max(y[m].sum(), 1e-9))
