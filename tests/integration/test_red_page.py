"""F10.6 — Integración de la página multilínea con streamlit.testing (AppTest) y coherencia con los artefactos."""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "plataforma"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (ART / "manifiesto.json").exists(), reason="faltan artefactos")]


@pytest.fixture
def at():
    a = AppTest.from_file(str(ROOT / "pages" / "2_Red_multilinea.py"), default_timeout=180)
    a.run()
    assert not a.exception, a.exception
    return a


def _table(at, i=0) -> pd.DataFrame:
    return at.table[i].value


def test_default_day_matches_artifacts(at):
    """La tabla del 28/08/2023 coincide con dias.csv.gz (categoría y score) para las 11 líneas."""
    man = json.loads((ART / "manifiesto.json").read_text(encoding="utf-8"))
    dias = pd.read_csv(ART / "dias.csv.gz", parse_dates=["fecha"])
    exp = dias[dias["fecha"] == "2023-08-28"].set_index("linea")
    T = _table(at)
    assert len(T) == len(exp) == 11
    for ln, row in exp.iterrows():
        shown = T.loc[ln.replace("LINEA ", ""), "Resultado"]
        assert man["categorias"][row["categoria"]] in shown
    assert "varias líneas" in T.loc["A", "Resultado"]  # IN3: la Línea A cae junto con otras líneas
    assert "anomalía estadística" in " ".join(w.value for w in at.warning)


def test_election_day_is_measurement_effect(at):
    at.date_input[0].set_value(dt.date(2023, 10, 29)).run()
    assert not at.exception
    T = _table(at)
    for ln in ("A", "B", "K", "T-A"):
        assert "Efecto de medición" in T.loc[ln, "Resultado"]
    assert "Efecto de medición" not in T.loc["1", "Resultado"]  # los buses sí cobran


def test_tabs_render_without_exceptions(at):
    labels = [t.label for t in at.tabs]
    assert any("Pronóstico" in x for x in labels) and any("Simulación" in x for x in labels)
    assert not at.exception
    # El catálogo muestra los estados de la evidencia, incluidas las hipótesis descartadas.
    cat = [t.value for t in at.table if "Estado" in t.value.columns]
    assert cat and (cat[0]["Estado"] == "descartado").sum() >= 2


def test_simulation_local_drop_is_local(at):
    at.selectbox(key="sim_line").set_value("LINEA A")
    at.date_input[0].set_value(dt.date(2024, 3, 6)).run()
    assert not at.exception
    txt = " ".join(m.value for m in at.markdown)
    assert "Alcance estimado para la Línea A: **local**" in txt
