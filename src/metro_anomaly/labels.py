"""Etiquetas heurísticas (IQR por día de la semana) — SOLO como métrica secundaria de consistencia.

No son verdad de terreno: son una regla estadística que no es causal (media móvil centrada y cuantiles
de todo el periodo) y que confunde efectos de calendario con anomalías (auditoría C5). Se mantienen:
- ``method="raw"``: comportamiento del notebook original (IQR sobre valores crudos; bug C1).
- ``method="resid"``: lo que el notebook decía hacer (IQR sobre los residuos de una descomposición
  multiplicativa con periodo 7), sin excluir la última hora por error (A1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from .calendar import day_type, is_holiday
from .data import SCORED_HOURS


def iqr_flags(values: pd.Series, groups, k: float = 1.5) -> pd.Series:
    out = pd.Series(0, index=values.index, dtype=int)
    for _, idx in values.groupby(np.asarray(groups)).groups.items():
        s = values.loc[idx].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out.loc[s.index] = ((s < q1 - k * iqr) | (s > q3 + k * iqr)).astype(int)
    return out


def heuristic_labels(W: pd.DataFrame, method: str = "resid", k: float = 1.5, hours=SCORED_HOURS) -> pd.DataFrame:
    """Devuelve banderas por hora y la etiqueta diaria ``any``.

    Celdas faltantes aisladas (≤ 1 % de las celdas) se interpolan en el tiempo SOLO para poder descomponer
    la serie (las etiquetas son una referencia offline, no causal) y nunca se marcan como atípicas.
    """
    nan_mask = W[hours].isna()
    if nan_mask.to_numpy().mean() > 0.01 or nan_mask.all(axis=1).any():
        raise ValueError("heuristic_labels requiere las horas evaluadas casi completas (≤ 1 % de NaN, sin días vacíos)")
    W = W.copy()
    W[hours] = W[hours].interpolate(limit_direction="both")
    dow = W.index.dayofweek
    flags = {}
    for h in hours:
        if method == "raw":
            v = W[h]
        elif method == "resid":
            v = seasonal_decompose(W[h], model="multiplicative", period=7).resid
        else:
            raise ValueError(method)
        flags[h] = iqr_flags(v, dow, k).where(~nan_mask[h], 0)
    F = pd.DataFrame(flags, index=W.index)
    if method == "resid":  # bordes sin residuo (media móvil centrada): sin etiqueta
        valid = seasonal_decompose(W[hours[0]], model="multiplicative", period=7).resid.notna()
        F = F.where(valid)
    F["n_horas"] = F[hours].sum(axis=1, min_count=1)
    F["any"] = (F["n_horas"] > 0).astype(float).where(F["n_horas"].notna())
    return F


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def label_quality(W: pd.DataFrame) -> dict:
    """Análisis de calidad: acuerdo entre variantes, sensibilidad a k y peso del calendario."""
    v = {
        "raw_k1.5": heuristic_labels(W, "raw", 1.5)["any"],
        "resid_k1.5": heuristic_labels(W, "resid", 1.5)["any"],
        "resid_k3.0": heuristic_labels(W, "resid", 3.0)["any"],
    }
    df = pd.DataFrame(v).dropna()
    fest = pd.Series(is_holiday(df.index), index=df.index)
    dtp = pd.Series(day_type(df.index), index=df.index)
    dec = pd.Series(df.index.month == 12, index=df.index)
    res = {"n_dias": len(df), "prevalencia": df.mean().round(3).to_dict()}
    names = list(v)
    res["kappa"] = {f"{a} vs {b}": round(cohen_kappa(df[a], df[b]), 3) for i, a in enumerate(names) for b in names[i + 1:]}
    res["tasa_en_festivos"] = df[fest].mean().round(3).to_dict()
    res["tasa_en_diciembre"] = df[dec].mean().round(3).to_dict()
    res["tasa_resto"] = df[~fest & ~dec].mean().round(3).to_dict()
    res["tasa_por_tipo_dia"] = df.groupby(dtp).mean().round(3).to_dict()
    return res
