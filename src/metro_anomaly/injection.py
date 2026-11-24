"""Catálogo pre-registrado de anomalías sintéticas (docs/PLAN_MAESTRO.md §5).

Cada escenario modifica una copia de días reales y devuelve la verdad de terreno: qué horas se alteraron
y si el escenario aplica a cada día. Los parámetros aleatorios (hora de inicio, duración) dependen solo
de la semilla, así que el resultado es determinista.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .calendar import day_type
from .data import SCORED_HOURS

H = {int(c[1:]): i for i, c in enumerate(SCORED_HOURS)}  # hora -> posición
NH = len(SCORED_HOURS)


@dataclass(frozen=True)
class Scenario:
    id: str
    tipo: str
    severidad: str
    descripcion: str
    apply: Callable  # (X: ndarray[n,NH], rng, ctx) -> (X', mask[n,NH], applicable[n])


def _consecutive(rng, n, lo, hi, kmin, kmax):
    """Bloques de k horas consecutivas con inicio en [lo, hi]."""
    ks = rng.integers(kmin, kmax + 1, size=n)
    starts = np.array([rng.integers(lo, hi - k + 2) for k in ks])
    mask = np.zeros((n, NH), bool)
    for i, (s, k) in enumerate(zip(starts, ks)):
        mask[i, [H[h] for h in range(s, s + k)]] = True
    return mask


def _block_factor(f, lo, hi, kmin, kmax):
    def apply(X, rng, ctx):
        m = _consecutive(rng, len(X), lo, hi, kmin, kmax)
        Y = X.copy()
        Y[m] = Y[m] * f
        return Y, m, np.ones(len(X), bool)
    return apply


def _day_factor(f):
    def apply(X, rng, ctx):
        return X * f, np.ones_like(X, bool), np.ones(len(X), bool)
    return apply


def _one_hour(op):
    def apply(X, rng, ctx):
        idx = rng.integers(H[6], H[20] + 1, size=len(X))
        m = np.zeros_like(X, bool)
        m[np.arange(len(X)), idx] = True
        Y = X.copy()
        Y[m] = op(Y[m])
        return Y, m, np.ones(len(X), bool)
    return apply


def _shift(X, rng, ctx):
    Y = X.copy()
    Y[:, 1:] = X[:, :-1]
    m = np.zeros_like(X, bool)
    m[:, 1:] = ~np.isclose(Y[:, 1:], X[:, 1:])
    return Y, m, np.ones(len(X), bool)


def _holiday_profile(X, rng, ctx):
    """Perfil mediano de domingo/festivo de las 8 semanas previas a cada día, aplicado a días hábiles."""
    dates, context = ctx["dates"], ctx["context"]
    dt_ctx = pd.Series(day_type(context.index), index=context.index)
    Y = X.copy()
    applicable = day_type(dates) == "habil"
    for i, t in enumerate(dates):
        if not applicable[i]:
            continue
        win = context.loc[t - pd.Timedelta(days=56): t - pd.Timedelta(days=1)]
        win = win[dt_ctx.loc[win.index] == "domingo_festivo"][SCORED_HOURS].dropna()
        if len(win) < 3:
            applicable[i] = False
            continue
        Y[i] = win.median().to_numpy()
    return Y, np.ones_like(X, bool) & applicable[:, None], applicable


SCENARIOS: list[Scenario] = [
    *[Scenario(f"T1_{s}", "T1", s, f"Interrupción: 1–3 h consecutivas (05–21) × {f}", _block_factor(f, 5, 21, 1, 3))
      for s, f in (("severa", 0.2), ("moderada", 0.5), ("leve", 0.7))],
    *[Scenario(f"T2_{s}", "T2", s, f"Todo el día × {f}", _day_factor(f))
      for s, f in (("severa", 0.6), ("moderada", 0.8), ("leve", 0.9))],
    *[Scenario(f"T3_{s}", "T3", s, f"Evento: 2–3 h (16–21) × {f}", _block_factor(f, 16, 21, 2, 3))
      for s, f in (("severa", 2.0), ("moderada", 1.5), ("leve", 1.25))],
    Scenario("T4_duplicada", "T4", "severa", "Error de datos: una hora (06–20) × 2", _one_hour(lambda v: v * 2)),
    Scenario("T4_cero", "T4", "severa", "Error de datos: una hora (06–20) = 0", _one_hour(lambda v: v * 0)),
    Scenario("T4_desfase", "T4", "severa", "Error de datos: perfil desfasado 1 h", _shift),
    Scenario("T5_perfil_festivo", "T5", "severa", "Perfil de domingo/festivo en un día hábil", _holiday_profile),
]
SCENARIO_BY_ID = {s.id: s for s in SCENARIOS}


def inject(X: pd.DataFrame, scenario: Scenario | str, seed: int, context: pd.DataFrame | None = None):
    """Aplica un escenario a los días de ``X`` (filas = fechas, columnas = SCORED_HOURS, conteos crudos).

    Devuelve ``(X_inyectado, mascara_horas, aplicable)``; ``X`` no se modifica.
    """
    sc = SCENARIO_BY_ID[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)
    arr = X[SCORED_HOURS].to_numpy(float)
    Y, m, ok = sc.apply(arr, rng, {"dates": X.index, "context": context})
    return (pd.DataFrame(Y, index=X.index, columns=SCORED_HOURS),
            pd.DataFrame(m, index=X.index, columns=SCORED_HOURS),
            pd.Series(ok, index=X.index))
