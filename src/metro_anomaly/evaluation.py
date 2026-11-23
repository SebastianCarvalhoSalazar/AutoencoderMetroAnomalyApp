"""Marco de evaluación: rolling origin mensual, inyección de anomalías, umbrales por presupuesto de
alertas, métricas y bootstrap por bloques semanales.

Protocolo (docs/PLAN_MAESTRO.md §2): para cada mes M del periodo evaluado, el detector se ajusta con
los días de ``train_start`` hasta el último día de M-1 y puntúa los días de M (reales e inyectados).
Ningún dato con fecha ≥ inicio de M entra en el ajuste. Dentro de M, la puntuación de cada día t
solo usa días < t (nivel, día de referencia).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .calendar import day_type, is_holiday
from .data import SCORED_HOURS
from .injection import SCENARIOS, inject

DEV = ("2022-01-01", "2023-06-30")
HOLDOUT = ("2023-07-01", "2024-06-30")
ALERT_BUDGET = 0.05


@dataclass
class RollingResult:
    detector: str
    real: pd.DataFrame  # index=fecha: score, tipo_dia, festivo, mes
    injected: pd.DataFrame  # filas: fecha, escenario, tipo, severidad, score, aplicable, top_hora_ok
    z_real: pd.DataFrame | None = None
    fit_seconds: float = 0.0
    meta: dict = field(default_factory=dict)


def month_starts(start, end):
    return pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), end, freq="MS")


def rolling_evaluate(factory: Callable, W: pd.DataFrame, eval_start, eval_end, train_start,
                     scenarios=SCENARIOS, seed: int = 0, train_end_cap=None) -> RollingResult:
    """``factory()`` -> Detector nuevo. ``W``: serie real completa (contexto)."""
    real_rows, inj_rows, zs = [], [], []
    det_name, fit_s = None, 0.0
    for ms in month_starts(eval_start, eval_end):
        me = min(ms + pd.offsets.MonthEnd(0), pd.Timestamp(eval_end))
        ms_eff = max(ms, pd.Timestamp(eval_start))
        hist_end = ms_eff - pd.Timedelta(days=1)
        if train_end_cap is not None:
            hist_end = min(hist_end, pd.Timestamp(train_end_cap))
        hist = W.loc[train_start:hist_end]
        assert hist.index.max() < ms_eff, "fuga: historia incluye el periodo evaluado"
        det = factory()
        det_name = det.name
        t0 = time.perf_counter()
        det.fit(hist, W)
        fit_s += time.perf_counter() - t0
        days = W.loc[ms_eff:me]
        days = days[days[SCORED_HOURS].notna().all(axis=1)]
        if days.empty:
            continue
        sc = det.score(days, W)
        real_rows.append(pd.DataFrame({"score": sc.day}))
        if sc.z is not None:
            zs.append(sc.z)
        for k, scn in enumerate(scenarios):
            s_seed = seed * 100_003 + int(ms.strftime("%Y%m")) * 31 + k
            Xi, mask, ok = inject(days, scn, s_seed, context=W)
            si = det.score(Xi, W)
            top_ok = np.full(len(days), np.nan)
            if si.z is not None:
                arg = si.z.abs().to_numpy()
                arg = np.where(np.isnan(arg), -np.inf, arg).argmax(1)
                top_ok = mask.to_numpy()[np.arange(len(days)), arg].astype(float)
            inj_rows.append(pd.DataFrame({"fecha": days.index, "escenario": scn.id, "tipo": scn.tipo,
                                          "severidad": scn.severidad, "score": si.day.to_numpy(),
                                          "aplicable": ok.to_numpy(), "top_hora_ok": top_ok}))
    real = pd.concat(real_rows)
    real["tipo_dia"] = day_type(real.index)
    real["festivo"] = is_holiday(real.index)
    real["mes"] = real.index.to_period("M").astype(str)
    return RollingResult(det_name, real, pd.concat(inj_rows, ignore_index=True),
                         pd.concat(zs) if zs else None, fit_s)


# --------------------------------------------------------------------------------------- umbrales
def budget_threshold(real_scores: pd.Series, budget: float = ALERT_BUDGET) -> float:
    """Umbral = cuantil (1-presupuesto) de los scores de días reales (fuera de muestra)."""
    s = real_scores.dropna().to_numpy()
    if len(s) == 0:
        raise ValueError("sin scores")
    return float(np.quantile(s, 1 - budget))


# --------------------------------------------------------------------------------------- métricas
def metrics(res: RollingResult, thr: float) -> dict:
    r = res.real
    alert = r["score"] > thr
    out = {"tasa_alertas": float(alert.mean()),
           "tasa_alertas_festivos": float(alert[r["festivo"]].mean()) if r["festivo"].any() else np.nan,
           "n_dias": int(len(r))}
    inj = res.injected[res.injected["aplicable"]]
    det = inj["score"] > thr
    for (sid), g in inj.groupby("escenario"):
        out[f"recall_{sid}"] = float((g["score"] > thr).mean())
    for sev, g in inj.groupby("severidad"):
        out[f"recall_{sev}"] = float((g["score"] > thr).mean())
    loc = inj[det & inj["tipo"].isin(["T1", "T3", "T4"])]["top_hora_ok"].dropna()
    out["localizacion_hora"] = float(loc.mean()) if len(loc) else np.nan
    mod = [k for k in out if k.startswith("recall_T") and k.endswith("_moderada")]
    out["recall_moderada_media"] = float(np.mean([out[k] for k in mod])) if mod else np.nan
    return out


def block_bootstrap(res: RollingResult, thr: float, keys, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """IC 95 % remuestreando semanas ISO completas (días reales e inyectados de esas semanas)."""
    rng = np.random.default_rng(seed)
    wk_real = res.real.index.to_period("W")
    wk_inj = pd.DatetimeIndex(res.injected["fecha"]).to_period("W")
    weeks = np.array(sorted(set(wk_real)))
    real_g = {w: res.real[wk_real == w] for w in weeks}
    inj_g = {w: res.injected[wk_inj == w] for w in weeks}
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(weeks, size=len(weeks), replace=True)
        rr = RollingResult(res.detector, pd.concat([real_g[w] for w in pick]), pd.concat([inj_g[w] for w in pick]))
        m = metrics(rr, thr)
        for k in keys:
            vals[k].append(m.get(k, np.nan))
    point = metrics(res, thr)
    return pd.DataFrame({k: {"valor": point.get(k), "ic_inf": np.nanpercentile(v, 2.5), "ic_sup": np.nanpercentile(v, 97.5)}
                         for k, v in vals.items()}).T


def paired_recall_diff(a: RollingResult, thr_a: float, b: RollingResult, thr_b: float, severidad="moderada",
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """Diferencia pareada de recall (a - b) sobre las mismas inyecciones, con IC por bootstrap semanal."""
    ia = a.injected[(a.injected.severidad == severidad) & a.injected.aplicable].reset_index(drop=True)
    ib = b.injected[(b.injected.severidad == severidad) & b.injected.aplicable].reset_index(drop=True)
    key = ["fecha", "escenario"]
    m = ia[key + ["score"]].merge(ib[key + ["score"]], on=key, suffixes=("_a", "_b"))
    m["d"] = (m["score_a"] > thr_a).astype(float) - (m["score_b"] > thr_b).astype(float)
    wk = pd.DatetimeIndex(m["fecha"]).to_period("W")
    weeks = np.array(sorted(set(wk)))
    groups = {w: m.loc[wk == w, "d"].to_numpy() for w in weeks}
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([groups[w] for w in rng.choice(weeks, len(weeks))]).mean() for _ in range(n_boot)]
    return {"diferencia": float(m["d"].mean()), "ic_inf": float(np.percentile(boots, 2.5)),
            "ic_sup": float(np.percentile(boots, 97.5)), "n": int(len(m))}
