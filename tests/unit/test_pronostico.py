import numpy as np
import pandas as pd
import pytest

from metro_anomaly.fuentes import clima
from metro_anomaly.pronostico import FEATURES_BASE, GBMForecaster, build_rows, wape
from tests.conftest import make_series


@pytest.fixture(scope="module")
def Ws():
    return {"A": make_series(seed=1), "B": make_series(seed=2) * 0.3}


def test_build_rows_is_causal(Ws):
    """Los rasgos del día t (nivel y rezago) no cambian si se alteran datos con fecha ≥ t."""
    dates = pd.date_range("2023-03-01", "2023-03-31")
    R = build_rows(Ws, dates)
    t = pd.Timestamp("2023-03-15")
    poisoned = {k: W.copy() for k, W in Ws.items()}
    for W in poisoned.values():
        W.loc[t:] *= 3
    P = build_rows(poisoned, dates)
    cols = ["nivel", "lag_mismo_tipo"]
    a = R[R["fecha"] == t][cols].reset_index(drop=True)
    b = P[P["fecha"] == t][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(R[R["fecha"] == t]["y_log"], P[P["fecha"] == t]["y_log"])  # el objetivo sí cambia


def test_gbm_intervals_ordered_and_reasonable(Ws):
    R = build_rows(Ws, pd.date_range("2022-02-01", "2023-06-30"))
    tr, te = R[R["fecha"] < "2023-06-01"], R[R["fecha"] >= "2023-06-01"]
    p = GBMForecaster(FEATURES_BASE, max_iter=60).fit(tr).predict(te)
    assert (p["p10"] <= p["pred"]).all() and (p["pred"] <= p["p90"]).all()
    assert wape(np.expm1(te["y_log"]), p["pred"]) < 0.08


def test_wape():
    assert wape([10, 10], [9, 11]) == pytest.approx(0.1)
    assert wape([10, np.nan], [9, 100]) == pytest.approx(0.1)


def test_clima_alignment(tmp_path):
    """La precipitación de Open-Meteo es la suma de la hora previa: la franja h toma el registro h+1."""
    t = pd.date_range("2023-01-01 00:00", periods=6, freq="h")
    pd.DataFrame({"tiempo_local": t, "precipitacion_mm_hora_previa": [0, 0, 0, 5.0, 0, 0],
                  "temperatura_c": 20.0}).to_csv(tmp_path / "c.csv", index=False)
    h = clima.load_hourly(tmp_path / "c.csv")
    assert h.loc["2023-01-01 02:00", "lluvia_mm"] == 5.0  # llovió entre 02:00 y 03:00
    assert h.loc["2023-01-01 03:00", "lluvia_mm"] == 0.0
