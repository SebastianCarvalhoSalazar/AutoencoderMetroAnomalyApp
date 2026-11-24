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


# ------------------------------------------------------------------------------------------ calibración e inferencia
CONFORMAL_GRID = np.linspace(0.5, 3.0, 51)


def conformal_factor(y, center, lo, hi, target=0.80) -> float:
    """Menor factor c de la rejilla tal que la banda [center − c·(center − lo), center + c·(hi − center)], en escala
    log1p, cubre ≥ ``target`` de ``y``. Todas las entradas en conteos (no log)."""
    ly, a, l, h = (np.log1p(np.asarray(v, float)) for v in (y, center, lo, hi))
    ok = ~(np.isnan(ly) | np.isnan(a) | np.isnan(l) | np.isnan(h))
    if not ok.any():
        return 1.0
    ly, a, l, h = ly[ok], a[ok], l[ok], h[ok]
    cov = np.array([((ly >= a - c * (a - l)) & (ly <= a + c * (h - a))).mean() for c in CONFORMAL_GRID])
    return float(CONFORMAL_GRID[np.argmax(cov >= target)] if cov.max() >= target else CONFORMAL_GRID[-1])


def conformal_online(P: pd.DataFrame, target=0.80) -> pd.DataFrame:
    """Calibración en línea (causal) del intervalo: para el mes M y cada línea, el factor c que ensancha o
    estrecha la banda del GBM (en escala log) se elige con las predicciones fuera de muestra de los meses < M.
    El primer mes queda sin calibrar (c = 1). Ajuste añadido DESPUÉS de ver que la cobertura de la Línea A
    no cumplía (docs/BITACORA.md); se evaluó en el mismo periodo, así que es una evidencia más débil."""
    P = P.copy()
    lp, l10, l90 = np.log1p(P["gbm"]), np.log1p(P["gbm_p10"]), np.log1p(P["gbm_p90"])
    P["_mes"] = P["fecha"].dt.to_period("M")
    P["c_conformal"] = 1.0
    for ln, g in P.groupby("linea"):
        for m in sorted(g["_mes"].unique()):
            prev = g[g["_mes"] < m]
            if prev.empty:
                continue
            c = conformal_factor(prev["y"], prev["gbm"], prev["gbm_p10"], prev["gbm_p90"], target)
            P.loc[(P["linea"] == ln) & (P["_mes"] == m), "c_conformal"] = c
    c = P["c_conformal"]
    P["gbm_p10c"] = np.expm1(lp - c * (lp - l10))
    P["gbm_p90c"] = np.expm1(lp + c * (l90 - lp))
    return P.drop(columns="_mes")


def forecast_day(Ws: dict, target, selection: dict, c_by_line: dict, train_start="2022-01-01") -> pd.DataFrame:
    """Pronóstico de ``target`` por línea y franja con el modelo elegido en DEV para cada línea (``selection``) y la
    banda del 80 % calibrada con ``c_by_line``. Solo usa datos con fecha < ``target`` (reajuste diario).

    Columnas: fecha, linea, hora, pred, p10, p90, modelo, ratio_p95_gbm (total diario del GBM base frente al p95
    de los 365 días previos; indicador H6)."""
    target = pd.Timestamp(target)
    last = target - pd.Timedelta(days=1)
    Wc = {ln: W.loc[:last] for ln, W in Ws.items()}
    R = build_rows(Wc, pd.date_range(train_start, target), include_future=True)
    tr, te = R[R["fecha"] <= last], R[R["fecha"] == target].copy()
    if te.empty:
        raise ValueError(f"Sin filas para pronosticar {target.date()}")
    for name, feats in (("gbm", FEATURES_BASE), ("gbm_v2", FEATURES_V2)):
        p = GBMForecaster(feats).fit(tr).predict(te)
        te[name], te[f"{name}_p10"], te[f"{name}_p90"] = p["pred"], p["p10"], p["p90"]
    prof = profile_predict(Wc, last, [target])
    te["perfil"] = [prof[ln].loc[target, f"h{h:02d}"] if ln in prof and target in prof[ln].index else np.nan
                    for ln, h in zip(te["linea"], te["hora"])]
    te["combo"] = np.expm1((np.log1p(te["gbm"]) + np.log1p(te["perfil"])) / 2)
    te["combo_v2"] = np.expm1((np.log1p(te["gbm_v2"]) + np.log1p(te["perfil"])) / 2)
    te["modelo"] = te["linea"].map(selection)
    te = te[te["modelo"].notna()].copy()
    te["pred"] = [te.at[i, m] for i, m in zip(te.index, te["modelo"])]
    c = te["linea"].map(c_by_line).fillna(1.0)
    lg, l10, l90, ls = np.log1p(te["gbm"]), np.log1p(te["gbm_p10"]), np.log1p(te["gbm_p90"]), np.log1p(te["pred"])
    # La banda del GBM se recentra en la predicción elegida (en log) y se escala con el factor conformal.
    te["p10"] = np.expm1(ls - c * (lg - l10))
    te["p90"] = np.expm1(ls + c * (l90 - lg))
    tot = te.groupby("linea")["gbm"].sum()
    p95 = {ln: Wc[ln].sum(axis=1, min_count=len(SCORED_HOURS)).loc[last - pd.Timedelta(days=364): last].quantile(0.95)
           for ln in tot.index}
    te["ratio_p95_gbm"] = te["linea"].map(lambda ln: tot[ln] / p95[ln])
    return te[["fecha", "linea", "hora", "pred", "p10", "p90", "modelo", "ratio_p95_gbm"]].reset_index(drop=True)
