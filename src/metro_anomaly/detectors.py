"""Detectores con interfaz común y garantía de causalidad.

Convenciones
------------
- ``W``: tabla diaria (índice = fechas contiguas, columnas = ``SCORED_HOURS``, conteos crudos, NaN = sin dato).
- ``fit(hist, context)``: ajusta con las filas de ``hist`` (todas anteriores al periodo evaluado).
  ``context`` es la serie real completa; de ella cada detector **solo** lee filas con fecha < t
  (nivel reciente, día de referencia), lo que hace la puntuación causal a resolución diaria.
- ``score(X, context)``: puntúa las filas de ``X``, que pueden ser reales o inyectadas.
  Devuelve un ``Scores`` con el score diario y, cuando aplica, los z por hora.

Nivel reciente (adaptación a cambios de demanda, auditoría A3)
-------------------------------------------------------------
``level(t)`` = mediana de log(Σ horas) de los días hábiles y sábados no festivos del intervalo
[t-28, t-1]. Los modelos de forma trabajan en log1p menos el nivel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

from .calendar import DAY_TYPES, calendar_features, day_type
from .data import SCORED_HOURS

LEVEL_WINDOW_DAYS = 28
LEVEL_MIN_DAYS = 10
SIGMA_FLOOR = 0.02
MAD_K = 1.4826


@dataclass
class Scores:
    day: pd.Series  # score diario (mayor = más anómalo)
    z: pd.DataFrame | None = None  # z firmados por hora (log-residuo estandarizado)


def _logtotal(W: pd.DataFrame) -> pd.Series:
    return np.log(W[SCORED_HOURS].sum(axis=1, min_count=len(SCORED_HOURS)))


def level_series(context: pd.DataFrame, dates) -> pd.Series:
    """Nivel causal para cada fecha de ``dates`` usando solo ``context`` con fecha < fecha."""
    dates = pd.DatetimeIndex(dates)
    lt = _logtotal(context)
    keep = np.isin(day_type(context.index), ["habil", "sabado"])
    lt = lt.where(keep)
    full_idx = pd.date_range(min(context.index.min(), dates.min()), max(context.index.max(), dates.max()), freq="D")
    lt = lt.reindex(full_idx)
    lvl = lt.rolling(f"{LEVEL_WINDOW_DAYS}D", closed="left", min_periods=LEVEL_MIN_DAYS).median()
    return lvl.reindex(dates)


def aggregate(z: pd.DataFrame, how: str = "max") -> pd.Series:
    a = z.abs()
    if how == "max":
        return a.max(axis=1, skipna=True)
    if how == "top2":
        return a.apply(lambda r: np.sort(r.dropna().to_numpy())[-2:].mean() if r.notna().sum() >= 2 else r.max(), axis=1)
    raise ValueError(how)


class Detector:
    name = "base"
    uses_z = True

    def __init__(self, agg: str = "max"):
        self.agg = agg

    def fit(self, hist: pd.DataFrame, context: pd.DataFrame) -> "Detector":
        raise NotImplementedError

    def score(self, X: pd.DataFrame, context: pd.DataFrame) -> Scores:
        raise NotImplementedError

    # utilidades comunes ------------------------------------------------------------------
    @staticmethod
    def _deleveled(X: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
        lvl = level_series(context, X.index)
        return np.log1p(X[SCORED_HOURS]).sub(lvl, axis=0)


class LevelShapeDetector(Detector):
    """Base de los modelos de forma: z = (log1p(x) - nivel(t) - g(t)) / sigma."""

    def _g(self, dates) -> pd.DataFrame:
        raise NotImplementedError

    def _sigma(self, dates) -> pd.DataFrame:
        raise NotImplementedError

    def expected_log(self, dates, context) -> pd.DataFrame:
        return self._g(dates).add(level_series(context, dates), axis=0)

    def score(self, X, context):
        e = self.expected_log(X.index, context)
        z = (np.log1p(X[SCORED_HOURS]) - e) / self._sigma(X.index)
        return Scores(aggregate(z, self.agg), z)

    def expected_counts(self, dates, context):
        """Banda esperada (en pasajeros): mediana y ±z·sigma, para la app."""
        e = self.expected_log(dates, context)
        return np.expm1(e), self._sigma(dates)


class ProfileDetector(LevelShapeDetector):
    """Perfil robusto por (tipo de día × hora): mediana y MAD de log1p - nivel en una ventana de historia."""

    name = "perfil_robusto"

    def __init__(self, agg="max", window_days=365, min_count=3):
        super().__init__(agg)
        self.window_days, self.min_count = window_days, min_count

    def fit(self, hist, context):
        hist = hist.loc[hist.index.max() - pd.Timedelta(days=self.window_days - 1):]
        Y = self._deleveled(hist, context).dropna(how="any")
        dt_ = pd.Series(day_type(Y.index), index=Y.index)
        self.med_, self.sig_ = {}, {}
        for t in DAY_TYPES:
            g = Y[dt_ == t]
            if len(g) >= self.min_count:
                self.med_[t] = g.median()
                self.sig_[t] = (MAD_K * (g - g.median()).abs().median()).clip(lower=SIGMA_FLOOR)
        fallback = {"especial": "sabado", "sabado": "habil", "domingo_festivo": "sabado", "habil": "sabado"}
        for t in DAY_TYPES:
            if t not in self.med_:
                self.med_[t], self.sig_[t] = self.med_[fallback[t]], self.sig_[fallback[t]] * 2
        return self

    def _rows(self, dates, table):
        return pd.DataFrame([table[t].to_numpy() for t in day_type(dates)], index=pd.DatetimeIndex(dates), columns=SCORED_HOURS)

    def _g(self, dates):
        return self._rows(dates, self.med_)

    def _sigma(self, dates):
        return self._rows(dates, self.sig_)


class RidgeCalendarDetector(LevelShapeDetector):
    """Regresión ridge multisalida: log1p - nivel ~ rasgos de calendario (tipo de día, festivo, vísperas,
    temporada de fin de año, Fourier anual). Sigma por hora = MAD del residuo de entrenamiento."""

    name = "regresion_calendario"

    def __init__(self, agg="max", alpha=1.0):
        super().__init__(agg)
        self.alpha = alpha

    @staticmethod
    def _features(dates):
        f = calendar_features(dates)
        dt_ = day_type(dates)
        for t in DAY_TYPES:
            f[f"tipo_{t}"] = (dt_ == t).astype(float)
        return f

    def fit(self, hist, context):
        Y = self._deleveled(hist, context).dropna(how="any")
        F = self._features(Y.index)
        self.model_ = Ridge(alpha=self.alpha).fit(F.to_numpy(), Y.to_numpy())
        R = Y.to_numpy() - self.model_.predict(F.to_numpy())
        self.sig_ = np.clip(MAD_K * np.median(np.abs(R - np.median(R, 0)), 0), SIGMA_FLOOR, None)
        return self

    def _g(self, dates):
        return pd.DataFrame(self.model_.predict(self._features(dates).to_numpy()), index=pd.DatetimeIndex(dates), columns=SCORED_HOURS)

    def _sigma(self, dates):
        return pd.DataFrame(np.tile(self.sig_, (len(dates), 1)), index=pd.DatetimeIndex(dates), columns=SCORED_HOURS)


class NaiveSeasonalDetector(Detector):
    """Línea base naive: valor esperado = último día anterior del mismo tipo (sin nivel ni modelo)."""

    name = "naive_semanal"

    def fit(self, hist, context):
        L = np.log1p(hist[SCORED_HOURS])
        ref = self._reference(hist.index, hist)
        D = (L - ref).dropna(how="any")
        self.sig_ = np.clip(MAD_K * (D - D.median()).abs().median().to_numpy(), SIGMA_FLOOR, None)
        return self

    @staticmethod
    def _reference(dates, context):
        """log1p del último día < t del mismo tipo, con datos completos (búsqueda hasta 35 días atrás)."""
        ctx = context[SCORED_HOURS].dropna(how="any")
        types = pd.Series(day_type(ctx.index), index=ctx.index)
        out = []
        for t, tt in zip(pd.DatetimeIndex(dates), day_type(dates)):
            prev = types.loc[t - pd.Timedelta(days=35): t - pd.Timedelta(days=1)]
            cand = prev.index[prev == tt]
            if len(cand) == 0 and tt == "especial":
                cand = prev.index[prev == "sabado"]
            out.append(np.log1p(ctx.loc[cand[-1]].to_numpy()) if len(cand) else np.full(len(SCORED_HOURS), np.nan))
        return pd.DataFrame(out, index=pd.DatetimeIndex(dates), columns=SCORED_HOURS)

    def score(self, X, context):
        z = (np.log1p(X[SCORED_HOURS]) - self._reference(X.index, context)) / self.sig_
        return Scores(aggregate(z, self.agg), z)


class IsolationForestDetector(Detector):
    """IsolationForest sobre log1p - nivel y tipo de día (sin explicación por hora)."""

    name = "isolation_forest"
    uses_z = False

    def __init__(self, agg="max", seed=0):
        super().__init__(agg)
        self.seed = seed

    def _X(self, X, context):
        Y = self._deleveled(X, context)
        dt_ = day_type(X.index)
        return np.hstack([Y.to_numpy(), np.stack([(dt_ == t).astype(float) for t in DAY_TYPES], 1)])

    def fit(self, hist, context):
        A = self._X(hist, context)
        A = A[~np.isnan(A).any(1)]
        self.model_ = IsolationForest(n_estimators=300, random_state=self.seed).fit(A)
        return self

    def score(self, X, context):
        A = self._X(X, context)
        s = np.full(len(A), np.nan)
        ok = ~np.isnan(A).any(1)
        if ok.any():
            s[ok] = -self.model_.score_samples(A[ok])
        return Scores(pd.Series(s, index=X.index))


REGISTRY = {
    "naive_semanal": NaiveSeasonalDetector,
    "perfil_robusto": ProfileDetector,
    "regresion_calendario": RidgeCalendarDetector,
    "isolation_forest": IsolationForestDetector,
}
