"""Integración con los 6 archivos oficiales reales (P1–P3)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metro_anomaly import calendar as C
from metro_anomaly import data as D

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def loaded():
    return D.load_all(ROOT / "data")


def test_all_files_load_and_validate(loaded):
    df, rep = loaded
    assert len(rep.files) == 6
    assert df["date"].min() == pd.Timestamp("2019-01-01") and df["date"].max() == pd.Timestamp("2024-06-30")
    assert not df.duplicated(["date", "line"]).any()
    assert "LINEA A" in set(df["line"]) and df["line"].nunique() >= 11


def test_known_cross_file_duplicate_resolved(loaded):
    df, rep = loaded
    assert rep.cross_file_duplicates == 1
    assert rep.dropped[0][0] == "2024-01-01" and rep.dropped[0][1] == "LINEA B"
    row = df[(df.date == "2024-01-01") & (df.line == "LINEA B")]
    assert len(row) == 1 and row["source"].iloc[0] == "Afluencia_Metro_2024.xlsx"


def test_total_equals_sum_of_hours(loaded):
    df, _ = loaded
    s = df[D.HOUR_COLS].sum(axis=1, min_count=1)
    assert np.isclose(s, df["total"]).mean() > 0.999


def test_line_a_coverage(loaded):
    df, _ = loaded
    w = D.line_series(df, "LINEA A")
    assert len(w) == 2008
    missing = w.index[w["missing_day"]]
    assert all(d.year == 2020 for d in missing)  # solo 2020 tiene días sin registro (363 de 366)
    # Hallazgo F2: horas faltantes dentro de días con registro (huecos de captura), solo en 2019 y 2021;
    # el periodo 2022–2024S1 está completo en 04–21. Se tratan como faltantes (no se imputan).
    gaps = w.drop(index=missing)[D.HOUR_COLS[:-2]].isna().any(axis=1)
    gap_days = w.drop(index=missing).index[gaps]
    assert len(gap_days) == 18 and set(gap_days.year) == {2019, 2021}
    assert w.loc["2022-01-01":, D.HOUR_COLS[:-2]].isna().sum().sum() == 0


def test_matches_legacy_pipeline_values(loaded):
    """Los valores no imputados coinciden con los usados por el pipeline original (2022–2024S1)."""
    df, _ = loaded
    w = D.line_series(df, "LINEA A", "2022-01-01", "2024-06-30")
    legacy = pd.read_excel(ROOT / "dev/results/full_2019_2024_data_linea_A_imputed.xlsx").set_index("Día")
    obs = w[D.HOUR_COLS[:-1]].notna().to_numpy()
    np.testing.assert_allclose(w[D.HOUR_COLS[:-1]].to_numpy()[obs], legacy[[f"{h:02d}:00:00" for h in D.HOURS[:-1]]].to_numpy()[obs])


def test_holiday_calendar_is_consistent_with_observed_profiles(loaded):
    """Contraste empírico (no oficial) del calendario: los festivos que caen entre semana deben parecerse
    más al perfil de domingo que al de día hábil del mismo año."""
    df, _ = loaded
    w = D.line_series(df, "LINEA A", "2022-01-01", "2024-06-30")
    X = np.log1p(w[D.SCORED_HOURS])
    fest = C.is_holiday(w.index)
    dow = w.index.dayofweek
    sunday = X[(dow == 6) & ~fest].median()
    weekday = X[(dow < 5) & ~fest].median()
    wk_fest = X[fest & (dow < 5)]
    closer_to_sunday = ((wk_fest - sunday).abs().mean(axis=1) < (wk_fest - weekday).abs().mean(axis=1)).mean()
    assert len(wk_fest) >= 30 and closer_to_sunday > 0.9
