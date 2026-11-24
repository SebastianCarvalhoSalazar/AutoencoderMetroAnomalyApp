import numpy as np
import pandas as pd
import pytest

from metro_anomaly.labels import cohen_kappa, heuristic_labels, iqr_flags


def test_iqr_flags_per_group():
    v = pd.Series([1, 2, 3, 4, 100, 10, 11, 12, 13, 14], dtype=float)
    g = [0] * 5 + [1] * 5
    assert iqr_flags(v, g).tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def test_sustained_level_shift_flagged_by_raw_not_by_resid(synth):
    """Mecanismo del bug C1: un periodo sostenido de menor nivel (como diciembre) se marca entero con
    IQR sobre valores crudos; con residuos (la tendencia absorbe el cambio) solo se marcan las transiciones."""
    W = synth.copy()
    block = W.index[200:240]
    W.loc[block] *= 0.5
    raw = heuristic_labels(W, "raw")["any"]
    res = heuristic_labels(W, "resid")["any"]
    inner = block[5:-5]
    # observado: crudo 0,83 frente a residuo 0,17 (≈ tasa base fuera del bloque, 0,20)
    assert raw.loc[inner].mean() > 0.7
    assert raw.loc[inner].mean() > 3 * res.loc[inner].mean()


def test_heuristic_missing_cells(synth):
    W = synth.copy(); W.iloc[10, 3] = np.nan
    F = heuristic_labels(W)
    assert F.iloc[10, 3] == 0  # celda faltante nunca se marca
    W.iloc[:, 0] = np.nan  # demasiados faltantes
    with pytest.raises(ValueError):
        heuristic_labels(W)


def test_kappa():
    assert cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == -1.0
