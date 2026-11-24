"""F5.2 — Integración de la app con streamlit.testing (AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


@pytest.fixture
def at():
    a = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    a.run()
    assert not a.exception, a.exception
    return a


def test_default_day_shows_anomaly(at):
    txt = " ".join(w.value for w in at.warning)
    assert "Anomalía estadística detectada" in txt and "18 h" in txt and "No es un incidente confirmado" in txt


def test_normal_day(at):
    at.date_input(key="fecha").set_value(__import__("datetime").date(2024, 3, 6)).run()
    assert not at.exception
    assert any("Sin anomalía estadística" in s.value for s in at.success)


def test_day_without_data_shows_message_not_trace(at):
    at.date_input(key="fecha").set_value(__import__("datetime").date(2020, 6, 13)).run()
    assert not at.exception
    assert any("No hay datos" in e.value for e in at.error)


def test_election_day_annotated(at):
    at.date_input(key="fecha").set_value(__import__("datetime").date(2023, 10, 29)).run()
    assert any("Elecciones" in m.value for m in at.info)


def test_out_of_range_typed_date_message(at):
    at.date_input(key="fecha").set_value(__import__("datetime").date(2018, 1, 1)).run()
    assert not at.exception
    assert any("fuera del rango" in e.value for e in at.error)


def test_metrics_and_severity_shown(at):
    labels = [m.label for m in at.metric]
    assert labels == ["Score del día", "Umbral", "Frecuencia histórica", "Desviación media del día"]
    assert any("severidad **extrema**" in w.value for w in at.warning)
