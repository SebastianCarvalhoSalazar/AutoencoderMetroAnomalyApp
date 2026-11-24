"""Análisis a nivel de red (F10, hipótesis H1, H2 y H4).

- ``network_z``: z por línea y hora con el perfil contextual de cada línea (ajustado de forma causal).
- ``CoherenceDetector`` (H1): residuo de la línea objetivo frente a lo que explica la red en la misma hora.
  Idea: lo que mueve a todas las líneas a la vez (lluvia, festivos mal clasificados, eventos de ciudad) no es
  una anomalía local; descontarlo debería dejar más visibles las anomalías propias de la línea.
- ``classify_scope`` (H2): local, multilínea o generalizada, según cuántas otras líneas se desvían en el mismo
  sentido y en las mismas horas.
- ``network_data_flags`` (H4): horas en que ≥ 3 líneas habituales no reportan (hueco o cero) a la vez,
  señal de posible falla de recaudo o de consolidación de datos a nivel de red.

Las reglas y los parámetros se fijaron antes de evaluar (docs/PLAN_MAESTRO.md §10).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from .data import SCORED_HOURS
from .detectors import Detector, ProfileContextDetector, Scores, aggregate, aggregation_stats

MAD_K = 1.4826
Z_DEV = 3.0  # |z| a partir del cual una línea "se desvía"
SCOPE_MULTI = 0.2  # fracción de otras líneas desviadas en el mismo sentido -> multilínea
SCOPE_GENERAL = 0.5  # -> generalizada (sistémica)


def fit_line_profiles(Ws: dict, train_start, hist_end) -> dict:
    """Perfil contextual por línea con historia [train_start, hist_end]; omite líneas sin días completos."""
    out = {}
    for ln, W in Ws.items():
        W = W[SCORED_HOURS]
        hist = W.loc[train_start:hist_end]
        try:
            out[ln] = ProfileContextDetector(agg="max").fit(hist, W.loc[:hist_end])
        except ValueError:
            continue
    return out


def network_z(Ws: dict, dets: dict, dates) -> dict:
    """z firmados por línea (fechas × horas) con los perfiles ``dets``. Días sin dato -> NaN."""
    dates = pd.DatetimeIndex(dates)
    out = {}
    for ln, det in dets.items():
        W = Ws[ln][SCORED_HOURS]
        X = W.reindex(dates)
        # Un cero es ausencia de registro (falla de datos u operación), no una señal de demanda: se excluye para que
        # no arrastre el índice de red (defecto hallado por tests/unit/test_plataforma.py; docs/BITACORA.md).
        out[ln] = det._z(X.where(X > 0), W)
    return out


def network_index(zs: dict, exclude=()) -> pd.DataFrame:
    """Mediana entre líneas del z (por fecha y hora), excluyendo ``exclude``."""
    arr = np.stack([z.to_numpy(float) for ln, z in zs.items() if ln not in exclude])
    idx = next(iter(zs.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # celdas sin ninguna línea -> NaN
        med = np.nanmedian(arr, axis=0)
    return pd.DataFrame(med, index=idx.index, columns=SCORED_HOURS)


class CoherenceDetector(Detector):
    """z_c(t,h) = (z_obj(t,h) − a_h − b_h · m(t,h)) / s_h, con m = mediana de z de las otras líneas."""

    name = "coherencia_red"

    def __init__(self, target: str, Ws: dict, train_start="2022-01-01", agg="peak_shift", horizon_days=70):
        super().__init__(agg)
        self.target, self.Ws, self.train_start, self.horizon_days = target, Ws, train_start, horizon_days

    def fit(self, hist, context):
        hist_end = hist.index.max()
        self.base_ = ProfileContextDetector(agg=self.agg).fit(hist, context)
        others = {ln: W for ln, W in self.Ws.items() if ln != self.target}
        self.dets_ = fit_line_profiles(others, self.train_start, hist_end)
        h = hist.loc[hist_end - pd.Timedelta(days=self.base_.window_days - 1):]
        h = h[h[SCORED_HOURS].notna().all(axis=1)]
        span = pd.date_range(h.index.min(), hist_end + pd.Timedelta(days=self.horizon_days))
        self.m_ = network_index(network_z(self.Ws, self.dets_, span))
        z_obj = self.base_._z(h, context)
        m = self.m_.reindex(h.index)
        self.a_, self.b_, self.s_ = {}, {}, {}
        for c in SCORED_HOURS:
            ok = z_obj[c].notna() & m[c].notna()
            x, y = m.loc[ok, c].clip(-6, 6).to_numpy()[:, None], z_obj.loc[ok, c].clip(-6, 6).to_numpy()
            if ok.sum() >= 30:
                hr = HuberRegressor().fit(x, y)
                a, b = float(hr.intercept_), float(hr.coef_[0])
            else:
                a, b = 0.0, 0.0
            r = z_obj.loc[ok, c] - a - b * m.loc[ok, c]
            s = MAD_K * (r - r.median()).abs().median() if ok.sum() >= 30 else 1.0
            self.a_[c], self.b_[c], self.s_[c] = a, b, max(float(s), 0.3)
        zc = self._zc(z_obj, m)
        self.agg_stats_ = aggregation_stats(zc.dropna(how="all"))
        return self

    def _zc(self, z_obj, m):
        mm = m.reindex(z_obj.index)
        pred = mm.mul(pd.Series(self.b_), axis=1).add(pd.Series(self.a_), axis=1).fillna(0.0)
        scale = pd.DataFrame(np.tile(pd.Series(self.s_)[SCORED_HOURS].to_numpy(), (len(z_obj), 1)),
                             index=z_obj.index, columns=SCORED_HOURS).where(mm.notna(), 1.0)
        return (z_obj - pred) / scale  # sin dato de red en (t, h): se usa el z propio

    def network_index_for(self, dates):
        missing = pd.DatetimeIndex(dates).difference(self.m_.index)
        if len(missing):
            self.m_ = pd.concat([self.m_, network_index(network_z(self.Ws, self.dets_, missing))]).sort_index()
        return self.m_.reindex(dates)

    def score(self, X, context):
        z_obj = self.base_._z(X, context)
        zc = self._zc(z_obj, self.network_index_for(X.index))
        return Scores(aggregate(zc, self.agg, self.agg_stats_), zc)


# ------------------------------------------------------------------------------------------ H2
def classify_scope(z_target: pd.Series, z_others: dict, z_thr: float = Z_DEV) -> dict:
    """Clasifica la desviación de un día de la línea objetivo.

    ``z_target``: z por hora (SCORED_HOURS) de la línea objetivo; ``z_others``: {línea: z por hora}.
    Horas consideradas: las de |z_target| > z_thr (o la de mayor |z| si ninguna). Una otra línea "acompaña"
    si su z medio en esas horas tiene el mismo signo y |z| > z_thr.
    """
    zt = z_target.astype(float)
    hrs = zt.index[zt.abs() > z_thr]
    if len(hrs) == 0:
        hrs = [zt.abs().idxmax()]
    sign = np.sign(zt[hrs].mean())
    acomp, evaluadas = [], 0
    for ln, z in z_others.items():
        v = z.reindex(hrs).astype(float)
        if v.notna().sum() == 0:
            continue
        evaluadas += 1
        mv = v.mean()
        if np.sign(mv) == sign and abs(mv) > z_thr:
            acomp.append(ln)
    frac = len(acomp) / evaluadas if evaluadas else np.nan
    if np.isnan(frac):
        alcance = "sin_red"
    elif frac >= SCOPE_GENERAL:
        alcance = "generalizada"
    elif frac >= SCOPE_MULTI:
        alcance = "multilinea"
    else:
        alcance = "local"
    return {"alcance": alcance, "fraccion_lineas": frac, "lineas_acompanan": acomp, "horas": [str(h) for h in hrs],
            "sentido": "caida" if sign < 0 else "aumento"}


# ------------------------------------------------------------------------------------------ H4
class NetworkDataMonitor:
    """Detecta horas con ≥ ``min_lines`` líneas habituales sin dato o en cero a la vez (posible falla de
    recaudo o de consolidación a nivel de red).

    Una línea es "habitual" en (t, h) si reportó esa hora en ≥ 90 % de los 28 días previos y la mediana de
    esos días es ≥ ``min_expected`` (todo con datos < t)."""

    def __init__(self, Ws: dict, start, end, min_lines: int = 3, min_expected: float = 100.0):
        self.min_lines, self.min_expected = min_lines, min_expected
        idx = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=28), end)
        self.full, self.cov, self.med = {}, {}, {}
        for ln, W in Ws.items():
            f = W[SCORED_HOURS].reindex(idx)
            self.full[ln] = f
            self.cov[ln] = f.notna().astype(float).rolling(28, min_periods=28).mean().shift(1)
            self.med[ln] = f.rolling(28, min_periods=10).median().shift(1)

    def habitual(self, t, h) -> list:
        return [ln for ln in self.full if self.cov[ln].at[t, h] >= 0.9 and self.med[ln].at[t, h] >= self.min_expected]

    def failing(self, t, h, override: dict | None = None) -> list:
        """Líneas habituales sin dato o en cero en (t, h). ``override``: {línea: valor} para simulaciones."""
        out = []
        for ln in self.habitual(t, h):
            v = (override or {}).get(ln, self.full[ln].at[t, h])
            if np.isnan(v) or v == 0:
                out.append(ln)
        return out

    def flags(self, dates) -> pd.DataFrame:
        rows = []
        for t in pd.DatetimeIndex(dates):
            for h in SCORED_HOURS:
                f = self.failing(t, h)
                if len(f) >= self.min_lines:
                    rows.append({"fecha": t, "hora": h, "n_lineas": len(f), "lineas": ",".join(sorted(f))})
        return pd.DataFrame(rows, columns=["fecha", "hora", "n_lineas", "lineas"])
