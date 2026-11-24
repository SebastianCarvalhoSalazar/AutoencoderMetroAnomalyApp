import numpy as np
import pandas as pd
import pytest

from metro_anomaly.detectors import ProfileDetector
from metro_anomaly.evaluation import (RollingResult, block_bootstrap, budget_threshold, metrics,
                                      paired_recall_diff, rolling_evaluate)


def test_budget_threshold():
    s = pd.Series(np.arange(100.0))
    assert budget_threshold(s, 0.05) == pytest.approx(94.05)
    with pytest.raises(ValueError):
        budget_threshold(pd.Series([np.nan]))


def _toy():
    idx = pd.date_range("2023-01-02", periods=14)
    real = pd.DataFrame({"score": np.r_[np.zeros(13), 10.0], "festivo": [False] * 13 + [True]}, index=idx)
    inj = pd.DataFrame({"fecha": list(idx[:4]) * 2, "escenario": ["T1_moderada"] * 4 + ["T1_severa"] * 4, "tipo": "T1",
                        "severidad": ["moderada"] * 4 + ["severa"] * 4, "score": [0, 5, 5, 5, 5, 5, 5, 5],
                        "aplicable": True, "top_hora_ok": [np.nan, 1, 0, 1, 1, 1, 1, 1]})
    return RollingResult("toy", real, inj)


def test_metrics_known_values():
    m = metrics(_toy(), thr=1.0)
    assert m["tasa_alertas"] == pytest.approx(1 / 14) and m["tasa_alertas_festivos"] == 1.0
    assert m["recall_T1_moderada"] == 0.75 and m["recall_severa"] == 1.0 and m["recall_moderada_media"] == 0.75
    assert m["localizacion_hora"] == pytest.approx(6 / 7)


def test_bootstrap_contains_point():
    b = block_bootstrap(_toy(), 1.0, ["recall_moderada"], n_boot=200)
    assert b.loc["recall_moderada", "ic_inf"] <= b.loc["recall_moderada", "valor"] <= b.loc["recall_moderada", "ic_sup"]


def test_paired_diff_zero_with_itself():
    t = _toy()
    d = paired_recall_diff(t, 1.0, t, 1.0, n_boot=50)
    assert d["diferencia"] == 0 and d["ic_inf"] == 0 and d["ic_sup"] == 0


def test_rolling_no_leakage_and_shapes(synth):
    seen = []

    class Spy(ProfileDetector):
        def fit(self, hist, context):
            seen.append(hist.index.max())
            return super().fit(hist, context)

    res = rolling_evaluate(Spy, synth, "2022-11-01", "2023-01-31", "2022-01-01")
    assert seen == [pd.Timestamp("2022-10-31"), pd.Timestamp("2022-11-30"), pd.Timestamp("2022-12-31")]
    assert len(res.real) == 92 and set(res.injected["escenario"].nunique() for _ in [0]) == {13}
    m = metrics(res, budget_threshold(res.real["score"]))
    assert 0.0 <= m["tasa_alertas"] <= 0.1


def test_metrics_excludes_nan_and_reports_coverage():
    t = _toy()
    t.real.loc[t.real.index[0], "score"] = np.nan
    m = metrics(t, 1.0)
    assert m["cobertura_real"] == pytest.approx(13 / 14) and m["tasa_alertas"] == pytest.approx(1 / 13)
