import numpy as np
import pandas as pd
import pytest

from metro_anomaly.calendar import day_type
from metro_anomaly.data import SCORED_HOURS


def make_series(start="2022-01-01", end="2023-06-30", seed=0, noise=0.03):
    """Serie sintética realista: perfil horario bimodal × tipo de día × ruido log-normal."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    hrs = np.array([int(c[1:]) for c in SCORED_HOURS])
    base = 20000 + 30000 * np.exp(-((hrs - 6.5) ** 2) / 2) + 35000 * np.exp(-((hrs - 17.5) ** 2) / 3)
    fac = {"habil": 1.0, "sabado": 0.7, "domingo_festivo": 0.4, "especial": 0.55}
    X = np.array([base * fac[t] for t in day_type(idx)]) * np.exp(rng.normal(0, noise, (len(idx), len(hrs))))
    return pd.DataFrame(np.round(X), index=idx, columns=SCORED_HOURS)


@pytest.fixture
def synth():
    return make_series()
