import numpy as np
import pandas as pd
import pytest

from metro_anomaly.data import SCORED_HOURS
from metro_anomaly.detectors import ProfileContextDetector
from metro_anomaly.red import CoherenceDetector, NetworkDataMonitor, classify_scope, fit_line_profiles, network_z

LINES = ["A", "B", "C", "D", "E"]


@pytest.fixture(scope="module")
def net(series_factory):
    """Cinco líneas con un factor común diario (choque de red) más ruido propio."""
    make_series = series_factory
    rng = np.random.default_rng(7)
    base = {ln: make_series(seed=i, noise=0.02) for i, ln in enumerate(LINES)}
    idx = base["A"].index
    common = np.exp(rng.normal(0, 0.06, (len(idx), len(SCORED_HOURS))))
    return {ln: (W * common).round() for ln, W in base.items()}


def _fit(net):
    hist = net["A"].loc[:"2023-01-31"]
    return CoherenceDetector("A", net, train_start="2022-01-01").fit(hist, net["A"])


def test_coherence_discounts_shared_shocks(net):
    det = _fit(net)
    t = pd.Timestamp("2023-02-15")
    shocked = {ln: W.copy() for ln, W in net.items()}
    for ln in LINES:  # choque sistémico: todas las líneas −40 % a las 17–18 h
        shocked[ln].loc[t, ["h17", "h18"]] *= 0.6
    det.Ws = shocked
    det.m_ = det.m_.drop(index=t)  # forzar recálculo del índice de red con los datos chocados
    X = shocked["A"].loc[[t]]
    s_coh = det.score(X, shocked["A"]).day.iloc[0]
    s_base = det.base_.score(X, shocked["A"]).day.iloc[0]
    assert s_base > 3 * s_coh


def test_coherence_keeps_local_shocks(net):
    det = _fit(net)
    t = pd.Timestamp("2023-02-15")
    X = net["A"].loc[[t]].copy()
    X[["h17", "h18"]] *= 0.6  # solo la línea A
    W = net["A"].copy(); W.loc[t] = X.iloc[0]
    assert det.score(X, W).day.iloc[0] > 5
    days = net["A"].loc["2023-02-01":"2023-02-28"]
    assert det.score(days, net["A"]).day.median() < 3


def test_coherence_is_causal(net):
    det = _fit(net)
    days = net["A"].loc["2023-02-01":"2023-02-28"]
    s_full = det.score(days, net["A"]).day
    t = days.index[10]
    poisoned = {ln: W.copy() for ln, W in net.items()}
    for W in poisoned.values():
        W.loc[t + pd.Timedelta(days=1):] *= 5
    det2 = _fit(poisoned)
    assert det2.score(days.loc[[t]], poisoned["A"]).day.iloc[0] == pytest.approx(s_full.loc[t])


def test_network_z_shapes(net):
    dets = fit_line_profiles(net, "2022-01-01", "2023-01-31")
    z = network_z(net, dets, pd.date_range("2023-02-01", "2023-02-05"))
    assert set(z) == set(LINES) and all(v.shape == (5, len(SCORED_HOURS)) for v in z.values())


def _z(vals: dict):
    s = pd.Series(0.0, index=SCORED_HOURS)
    for h, v in vals.items():
        s[h] = v
    return s


def test_classify_scope():
    zt = _z({"h17": -8, "h18": -10})
    others_local = {ln: _z({}) for ln in "BCDEFGHIJK"}
    assert classify_scope(zt, others_local)["alcance"] == "local"
    others_multi = {**others_local, "B": _z({"h17": -6, "h18": -6}), "C": _z({"h18": -9})}
    r = classify_scope(zt, others_multi)
    assert r["alcance"] == "multilinea" and r["lineas_acompanan"] == ["B", "C"]
    others_gen = {ln: _z({"h17": -5, "h18": -5}) for ln in "BCDEFGHIJK"}
    assert classify_scope(zt, others_gen)["alcance"] == "generalizada"
    # sentido opuesto (redistribución) no cuenta como acompañamiento
    others_opp = {ln: _z({"h17": 6, "h18": 6}) for ln in "BCDEFGHIJK"}
    assert classify_scope(zt, others_opp)["alcance"] == "local"


def test_network_data_monitor(net):
    mon = NetworkDataMonitor(net, "2023-02-01", "2023-02-28")
    t, h = pd.Timestamp("2023-02-10"), "h08"
    assert mon.failing(t, h) == []
    assert len(mon.failing(t, h, {"A": 0.0, "B": np.nan, "C": 0.0})) == 3
    assert len(mon.failing(t, h, {"A": 0.0, "B": np.nan})) == 2
    assert mon.flags(pd.date_range("2023-02-01", "2023-02-28")).empty
