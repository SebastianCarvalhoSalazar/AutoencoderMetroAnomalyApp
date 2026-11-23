import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")
from metro_anomaly.detectors_nn import REGISTRY_NN  # noqa: E402
from metro_anomaly.injection import inject  # noqa: E402

pytestmark = pytest.mark.research


@pytest.fixture(scope="module")
def fitted():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conftest import make_series
    s = make_series()
    return s, {n: (cls(seed=0).fit(s.loc[:"2023-01-31"], s)) for n, cls in REGISTRY_NN.items()}


@pytest.mark.parametrize("name", list(REGISTRY_NN))
def test_causality_and_severe_detection(fitted, name):
    s, dets = fitted
    det = dets[name]
    days = s.loc["2023-02-01":"2023-02-28"]
    full = det.score(days, s).day
    t = days.index[10]
    poisoned = s.copy(); poisoned.loc[t + pd.Timedelta(days=1):] *= 5
    assert det.score(days.loc[[t]], poisoned).day.iloc[0] == pytest.approx(full.loc[t], rel=1e-6)
    inj = det.score(inject(days, "T1_severa", 0, s)[0], s).day
    assert (inj.to_numpy() > full.to_numpy()).mean() > 0.8


@pytest.mark.parametrize("name", list(REGISTRY_NN))
def test_determinism_with_seed(fitted, name):
    s, dets = fitted
    days = s.loc["2023-02-01":"2023-02-10"]
    again = REGISTRY_NN[name](seed=0).fit(s.loc[:"2023-01-31"], s)
    pd.testing.assert_series_equal(dets[name].score(days, s).day, again.score(days, s).day)
