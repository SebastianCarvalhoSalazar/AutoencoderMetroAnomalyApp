import numpy as np
import pandas as pd
import pytest

from metro_anomaly.data import SCORED_HOURS
from metro_anomaly.detectors import REGISTRY, aggregate, level_series
from metro_anomaly.injection import inject

NAMES = list(REGISTRY)


def test_level_is_causal_and_correct(synth):
    t = pd.Timestamp("2023-02-15")
    full = level_series(synth, [t]).iloc[0]
    trunc = level_series(synth.loc[:t - pd.Timedelta(days=1)], [t]).iloc[0]
    assert full == trunc
    poisoned = synth.copy(); poisoned.loc[t:] *= 10  # el futuro (incluido t) no debe afectar
    assert level_series(poisoned, [t]).iloc[0] == full


@pytest.mark.parametrize("name", NAMES)
def test_causality_D8(synth, name):
    """D8: el score de un día no cambia si se alteran o eliminan datos posteriores a ese día."""
    hist = synth.loc[:"2023-01-31"]
    days = synth.loc["2023-02-01":"2023-02-28"]
    det = REGISTRY[name]().fit(hist, synth)
    s_full = det.score(days, synth).day
    for t in days.index[[3, 15, 27]]:
        poisoned = synth.copy(); poisoned.loc[t + pd.Timedelta(days=1):] *= 5
        truncated = synth.loc[:t]
        assert det.score(days.loc[[t]], poisoned).day.iloc[0] == s_full.loc[t]
        assert det.score(days.loc[[t]], truncated).day.iloc[0] == s_full.loc[t]


@pytest.mark.parametrize("name", NAMES)
def test_determinism_D9(synth, name):
    hist, days = synth.loc[:"2023-01-31"], synth.loc["2023-02-01":"2023-02-28"]
    a = REGISTRY[name]().fit(hist, synth).score(days, synth).day
    b = REGISTRY[name]().fit(hist, synth).score(days, synth).day
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_severe_injection_scores_higher(synth, name):
    hist, days = synth.loc[:"2023-01-31"], synth.loc["2023-02-01":"2023-02-28"]
    det = REGISTRY[name]().fit(hist, synth)
    real = det.score(days, synth).day
    inj = det.score(inject(days, "T1_severa", 0, synth)[0], synth).day
    assert (inj.to_numpy() > real.to_numpy()).mean() > 0.9


@pytest.mark.parametrize("name", ["perfil_robusto", "regresion_calendario", "naive_semanal"])
def test_z_sign_and_localization(synth, name):
    hist, days = synth.loc[:"2023-01-31"], synth.loc["2023-02-01":"2023-02-28"]
    det = REGISTRY[name]().fit(hist, synth)
    Xi, m, _ = inject(days, "T1_severa", 0, synth)
    z = det.score(Xi, synth).z
    arg = z.abs().to_numpy().argmax(1)
    assert m.to_numpy()[np.arange(len(days)), arg].mean() > 0.9
    assert (z.to_numpy()[m.to_numpy()] < 0).mean() > 0.95  # caídas -> z negativos


def test_missing_hour_is_ignored_not_crash(synth):
    hist, days = synth.loc[:"2023-01-31"], synth.loc["2023-02-01":"2023-02-03"].copy()
    days.iloc[0, 3] = np.nan
    det = REGISTRY["perfil_robusto"]().fit(hist, synth)
    s = det.score(days, synth)
    assert np.isfinite(s.day).all() and np.isnan(s.z.iloc[0, 3])


def test_aggregate():
    z = pd.DataFrame([[1.0, -4.0, 2.0], [np.nan, 1.0, -3.0]])
    assert aggregate(z, "max").tolist() == [4.0, 3.0]
    assert aggregate(z, "top2").tolist() == [3.0, 2.0]
