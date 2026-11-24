import numpy as np
import pandas as pd
import pytest

from metro_anomaly.fuentes import clima
from metro_anomaly.pronostico import FEATURES_BASE, GBMForecaster, build_rows, wape


@pytest.fixture(scope="module")
def Ws(series_factory):
    return {"A": series_factory(seed=1), "B": series_factory(seed=2) * 0.3}


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


def test_conformal_factor_reaches_target():
    from metro_anomaly.pronostico import conformal_factor
    rng = np.random.default_rng(0)
    center = np.full(2000, 1000.0)
    y = np.expm1(np.log1p(center) + rng.normal(0, 0.1, 2000))
    lo, hi = np.expm1(np.log1p(center) - 0.05), np.expm1(np.log1p(center) + 0.05)  # banda demasiado estrecha
    c = conformal_factor(y, center, lo, hi, target=0.8)
    ly, a = np.log1p(y), np.log1p(center)
    cov = ((ly >= a - c * 0.05) & (ly <= a + c * 0.05)).mean()
    assert c > 1 and 0.8 <= cov <= 0.83
    assert conformal_factor([np.nan], [1], [1], [1]) == 1.0


def test_forecast_day_is_causal(Ws):
    """El pronóstico del día t no cambia si se alteran datos con fecha ≥ t."""
    from metro_anomaly.pronostico import forecast_day
    sel, c = {"A": "combo_v2", "B": "gbm"}, {"A": 1.2, "B": 1.0}
    t = pd.Timestamp("2023-06-15")
    Wc = {k: W.loc[:"2023-06-30"] for k, W in Ws.items()}
    f1 = forecast_day(Wc, t, sel, c)
    poisoned = {k: W.copy() for k, W in Wc.items()}
    for W in poisoned.values():
        W.loc[t:] *= 5
    f2 = forecast_day(poisoned, t, sel, c)
    pd.testing.assert_frame_equal(f1, f2)
    assert len(f1) == 2 * 19 and (f1["p10"] <= f1["pred"]).all() and (f1["pred"] <= f1["p90"]).all()
    assert set(f1["modelo"]) == {"combo_v2", "gbm"}


def test_forecast_day_closure_is_zero_and_not_contaminating(Ws):
    from metro_anomaly.pronostico import forecast_day
    sel, c = {"A": "combo_v2", "B": "gbm"}, {"A": 1.0, "B": 1.0}
    Wc = {k: W.loc[:"2023-06-30"].copy() for k, W in Ws.items()}
    t = pd.Timestamp("2023-06-15")
    f = forecast_day(Wc, t, sel, c, cierres={("B", t)})
    b = f[f["linea"] == "B"]
    assert (b[["pred", "p10", "p90"]] == 0).all().all() and set(b["modelo"]) == {"cierre_programado"}
    # Días de cierre pasados con registro residual no arrastran el pronóstico del día siguiente.
    Wc["B"].loc["2023-06-12":"2023-06-14"] = 5.0
    clo = {("B", pd.Timestamp(d)) for d in pd.date_range("2023-06-12", "2023-06-14")}
    g_mask = forecast_day(Wc, t, sel, c, cierres=clo)
    g_raw = forecast_day(Wc, t, sel, c)
    ref = forecast_day({k: W.loc[:"2023-06-30"] for k, W in Ws.items()}, t, sel, c)
    err = lambda g: float((g[g.linea == "B"]["pred"].to_numpy() / ref[ref.linea == "B"]["pred"].to_numpy() - 1).__abs__().mean())
    assert err(g_mask) < 0.05 < err(g_raw)
