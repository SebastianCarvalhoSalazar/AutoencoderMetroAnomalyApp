"""Página de investigación (rama de autoencoders): se dibuja sin errores y coincide con los resultados exportados."""
import datetime as dt
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "reports" / "investigacion_ae" / "tablero"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (TAB / "progreso.json").exists(), reason="faltan resultados")]


@pytest.fixture
def at():
    a = AppTest.from_file(str(ROOT / "pages" / "3_Investigacion_modelos.py"), default_timeout=120)
    a.run()
    assert not a.exception, a.exception
    return a


def test_verdicts_table_matches_plan(at):
    prog = json.loads((TAB / "progreso.json").read_text(encoding="utf-8"))["rows"]
    T = at.table[0].value
    assert len(T) == len(prog) == 8
    assert "NO CUMPLE" in T.loc["AE.2", "Resultado"] and "CUMPLE" in T.loc["AE.5", "Resultado"]


@pytest.mark.parametrize("ej", ["normal", "caida_horas", "caida_dia", "hora_duplicada", "incidente_real"])
def test_each_example_renders_with_its_decision(at, ej):
    e = {r["ejemplo"]: r for r in json.loads((TAB / "ex_ejemplos.json").read_text(encoding="utf-8"))["rows"]}[ej]
    at.selectbox(key="ej").set_value(ej).run()
    assert not at.exception
    txt = " ".join(m.value for m in at.markdown)
    assert ("**alerta**" if e["alerta_ae"] else "**sin alerta**") in txt.split("score del autoencoder")[1]
    assert ("**alerta**" if e["alerta_perfil"] else "**sin alerta**") in txt.split("Este día: score")[1]


def test_coherence_section_handles_dates(at):
    at.selectbox(key="coh_ln").set_value("LINEA B").run()
    at.date_input(key="coh_f").set_value(dt.date(2023, 10, 29)).run()
    assert not at.exception
    at.date_input(key="coh_f").set_value(dt.date(2022, 4, 1)).run()
    assert not at.exception
