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
    assert any("ANOMALÍA" in e.value for e in at.error)
    assert "18h" in " ".join(e.value for e in at.error)


def test_normal_day(at):
    at.date_input(key="fecha").set_value(__import__("datetime").date(2024, 3, 6)).run()
    assert not at.exception
    assert any("Normal" in s.value for s in at.success)


def test_day_without_data_shows_message_not_trace(at):
    at.date_input(key="fecha").set_value(__import__("datetime").date(2020, 6, 13)).run()
    assert not at.exception
    assert any("No hay datos" in e.value for e in at.error)


def test_election_day_annotated(at):
    at.date_input(key="fecha").set_value(__import__("datetime").date(2023, 10, 29)).run()
    assert any("Elecciones" in m.value for m in at.markdown)
