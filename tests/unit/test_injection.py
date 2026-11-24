import numpy as np
import pandas as pd
import pytest

from metro_anomaly.calendar import day_type
from metro_anomaly.data import SCORED_HOURS
from metro_anomaly.injection import H, SCENARIO_BY_ID, SCENARIOS, inject


@pytest.fixture
def days(synth):
    return synth.loc["2023-03-01":"2023-03-31"]


@pytest.mark.parametrize("sid,f", [("T1_severa", 0.2), ("T1_moderada", 0.5), ("T1_leve", 0.7), ("T3_severa", 2.0), ("T3_moderada", 1.5)])
def test_block_factor_exact_and_consecutive(days, synth, sid, f):
    Y, m, ok = inject(days, sid, seed=1, context=synth)
    M = m.to_numpy()
    assert ok.all() and (M.sum(1) >= 1).all() and (M.sum(1) <= 3).all()
    np.testing.assert_allclose(Y.to_numpy()[M], days.to_numpy()[M] * f)
    np.testing.assert_allclose(Y.to_numpy()[~M], days.to_numpy()[~M])
    for row in M:  # bloque consecutivo
        pos = np.flatnonzero(row)
        assert (np.diff(pos) == 1).all()
    if sid.startswith("T3"):
        assert not M[:, : H[16]].any()


def test_day_factor(days, synth):
    Y, m, ok = inject(days, "T2_moderada", seed=0, context=synth)
    np.testing.assert_allclose(Y.to_numpy(), days.to_numpy() * 0.8)
    assert m.to_numpy().all()


def test_data_errors(days, synth):
    Y, m, _ = inject(days, "T4_cero", seed=3, context=synth)
    assert (Y.to_numpy()[m.to_numpy()] == 0).all() and m.to_numpy().sum(1).tolist() == [1] * len(days)
    Y, m, _ = inject(days, "T4_duplicada", seed=3, context=synth)
    np.testing.assert_allclose(Y.to_numpy()[m.to_numpy()], days.to_numpy()[m.to_numpy()] * 2)
    Y, _, _ = inject(days, "T4_desfase", seed=3, context=synth)
    np.testing.assert_allclose(Y.to_numpy()[:, 1:], days.to_numpy()[:, :-1])


def test_holiday_profile_only_on_weekdays_and_uses_past(days, synth):
    Y, m, ok = inject(days, "T5_perfil_festivo", seed=0, context=synth)
    assert (ok.to_numpy() == (day_type(days.index) == "habil")).all()
    t = days.index[ok.to_numpy()][0]
    win = synth.loc[t - pd.Timedelta(days=56): t - pd.Timedelta(days=1)]
    win = win[day_type(win.index) == "domingo_festivo"]
    np.testing.assert_allclose(Y.loc[t].to_numpy(), win.median().to_numpy())


def test_deterministic_and_input_untouched(days, synth):
    before = days.copy()
    a = inject(days, "T1_moderada", seed=7, context=synth)[0]
    b = inject(days, "T1_moderada", seed=7, context=synth)[0]
    c = inject(days, "T1_moderada", seed=8, context=synth)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    pd.testing.assert_frame_equal(days, before)


def test_catalog_matches_plan():
    assert {s.id for s in SCENARIOS} == {"T1_severa", "T1_moderada", "T1_leve", "T2_severa", "T2_moderada", "T2_leve",
                                         "T3_severa", "T3_moderada", "T3_leve", "T4_duplicada", "T4_cero", "T4_desfase",
                                         "T5_perfil_festivo"}
    assert SCENARIO_BY_ID["T2_severa"].severidad == "severa"
