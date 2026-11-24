import numpy as np
import pandas as pd
import pytest

from metro_anomaly import plataforma as PL
from tests.conftest import make_series

LINES = {"LINEA A": "metro", "LINEA B": "metro", "LINEA K": "cable", "LINEA 1": "bus", "LINEA 2": "bus"}


@pytest.fixture(scope="module")
def evaluated():
    Ws = {ln: make_series(start="2022-01-01", end="2022-04-30", seed=i, noise=0.02) for i, ln in enumerate(LINES)}
    Ws["LINEA A"].loc["2022-03-16", "h10"] = np.nan  # ausencia simultánea en 3 líneas
    Ws["LINEA B"].loc["2022-03-16", "h10"] = 0
    Ws["LINEA K"].loc["2022-03-16", "h10"] = np.nan
    Ws["LINEA A"].loc["2022-03-17", "h10"] = 0  # cero en una sola línea
    Ws["LINEA A"].loc["2022-03-22", ["h17", "h18"]] *= 0.3  # caída local fuerte
    for ln in LINES:  # caída generalizada
        Ws[ln].loc["2022-03-24", ["h07", "h08"]] *= 0.3
    red = PL.Red(Ws, pd.DataFrame({"linea": list(LINES), "modo": list(LINES.values())}),
                 pd.DataFrame(columns=["linea_origen", "linea_destino", "estacion_transferencia"]))
    horas, dias = PL.build_history(red, "2022-03-01", "2022-03-31")
    thr = {ln: {"score": "score_coherencia" if ln in PL.COHERENCE_LINES else "score_perfil", "umbral": 6.0} for ln in LINES}
    ev = PL.evaluate_days(red, horas, dias, thr).set_index(["fecha", "linea"])
    return horas, dias, ev


def cat(ev, d, ln):
    return ev.loc[(pd.Timestamp(d), ln), "categoria"]


def test_election_is_measurement_effect_only_for_fare_free_modes(evaluated):
    _, _, ev = evaluated
    assert cat(ev, "2022-03-13", "LINEA A") == "efecto_de_medicion"
    assert cat(ev, "2022-03-13", "LINEA K") == "efecto_de_medicion"
    assert cat(ev, "2022-03-13", "LINEA 1") != "efecto_de_medicion"


def test_simultaneous_absence_and_single_line_data_problem(evaluated):
    _, _, ev = evaluated
    for ln in ("LINEA A", "LINEA B", "LINEA K"):
        assert cat(ev, "2022-03-16", ln) == "ausencia_simultanea_red"
    assert cat(ev, "2022-03-16", "LINEA 1") == "sin_anomalia"
    assert cat(ev, "2022-03-17", "LINEA A") == "problema_de_datos"


def test_local_and_generalized_anomalies(evaluated):
    _, _, ev = evaluated
    assert cat(ev, "2022-03-22", "LINEA A") == "anomalia_local"
    assert ev.loc[(pd.Timestamp("2022-03-22"), "LINEA A"), "horas_fuera"] == "17,18"
    assert cat(ev, "2022-03-24", "LINEA 2") == "anomalia_generalizada"


def test_normal_days_mostly_clean(evaluated):
    _, _, ev = evaluated
    assert (ev["categoria"] == "sin_anomalia").mean() > 0.8


def test_history_is_causal(evaluated):
    """Los z de marzo no cambian si se alteran los datos de abril en adelante."""
    horas, _, _ = evaluated
    Ws = {ln: make_series(start="2022-01-01", end="2022-04-30", seed=i, noise=0.02) for i, ln in enumerate(LINES)}
    red = PL.Red(Ws, pd.DataFrame({"linea": list(LINES), "modo": list(LINES.values())}), pd.DataFrame())
    h1, _ = PL.build_history(red, "2022-03-01", "2022-03-10")
    for W in Ws.values():
        W.loc["2022-03-11":] *= 4
    h2, _ = PL.build_history(red, "2022-03-01", "2022-03-10")
    pd.testing.assert_series_equal(h1["z"], h2["z"])


def test_catalog_and_data_kinds_complete():
    assert set(PL.TIPOS_DATO) == {"real", "estimacion", "simulacion", "prediccion"}
    assert {m["estado"].split(" ")[0] for m in PL.CATALOGO_MODELOS} >= {"adoptado", "descriptivo", "descartado"}
