import datetime as dt

import numpy as np
import pandas as pd
import pytest

from metro_anomaly import data as D


def _raw(n_days=3, line="Línea A", start="01/01/2023", time_cols=True):
    dates = pd.date_range(pd.to_datetime(start, dayfirst=True), periods=n_days).strftime("%d/%m/%Y")
    cols = [dt.time(h) if time_cols else f"{h:02d}:00:00" for h in D.HOURS]
    df = pd.DataFrame({"Día": dates, "Línea de Servicio": line})
    for i, c in enumerate(cols):
        df[c] = float(100 + i)
    df[D.TOTAL_COL] = df[cols].sum(axis=1)
    return df


@pytest.mark.parametrize("time_cols", [True, False])
def test_standardize_ok(time_cols):
    out = D.standardize(_raw(time_cols=time_cols))
    assert list(out.columns) == ["date", "line", *D.HOUR_COLS, "total", "source"]
    assert out["line"].unique().tolist() == ["LINEA A"]
    assert out["date"].iloc[0] == pd.Timestamp("2023-01-01")
    assert out["h04"].iloc[0] == 100.0 and out["h23"].iloc[0] == 119.0


def test_dates_are_day_first_not_month_first():
    out = D.standardize(_raw(start="02/01/2023", n_days=1))
    assert out["date"].iloc[0] == pd.Timestamp("2023-01-02")


@pytest.mark.parametrize("name,expected", [("Línea a ", "LINEA A"), ("LÍNEA  T-A", "LINEA T-A"), ("linea b", "LINEA B")])
def test_normalize_line(name, expected):
    assert D.normalize_line(name) == expected


def test_missing_hour_column_rejected():
    r = _raw().drop(columns=[dt.time(10)])
    with pytest.raises(D.DataValidationError, match="faltan \\[10\\]"):
        D.standardize(r)


def test_missing_required_column_rejected():
    with pytest.raises(D.DataValidationError, match="obligatorias"):
        D.standardize(_raw().drop(columns=["Día"]))


def test_negative_and_non_numeric_rejected():
    r = _raw(); r[dt.time(5)] = r[dt.time(5)].astype(object); r.loc[0, dt.time(5)] = "abc"
    with pytest.raises(D.DataValidationError, match="no numéricos"):
        D.standardize(r)
    r = _raw(); r.loc[1, dt.time(6)] = -1
    with pytest.raises(D.DataValidationError, match="negativos"):
        D.standardize(r)


def test_bad_date_rejected():
    r = _raw(); r.loc[0, "Día"] = "31/02/2023"
    with pytest.raises(D.DataValidationError, match="Fechas"):
        D.standardize(r)


def test_nan_hours_kept_not_imputed():
    r = _raw(); r.loc[0, dt.time(23)] = np.nan
    out = D.standardize(r)
    assert np.isnan(out["h23"].iloc[0])


def test_within_file_duplicates_rejected():
    r = pd.concat([_raw(), _raw()])
    with pytest.raises(D.DataValidationError, match="duplicadas"):
        D.standardize(r)


def test_cross_file_duplicate_resolution_prefers_own_year():
    a = D.standardize(_raw(start="01/01/2024", n_days=1), source="Afluencia_Metro_2023.xlsx")
    b = D.standardize(_raw(start="01/01/2024", n_days=1), source="Afluencia_Metro_2024.xlsx")
    b.loc[:, "h10"] = 999.0
    rep = D.LoadReport()
    out = D.resolve_duplicates(pd.concat([a, b]), rep)
    assert len(out) == 1 and out["source"].iloc[0] == "Afluencia_Metro_2024.xlsx" and out["h10"].iloc[0] == 999.0
    assert rep.cross_file_duplicates == 1 and rep.conflicting_duplicates == 1


def test_line_series_reindexes_missing_days():
    d = D.standardize(_raw(n_days=5)).drop(index=[2])
    w = D.line_series(d, "linea a")
    assert len(w) == 5 and w["missing_day"].sum() == 1 and w.loc["2023-01-03", D.HOUR_COLS].isna().all()


def test_line_series_unknown_line():
    with pytest.raises(D.DataValidationError):
        D.line_series(D.standardize(_raw()), "LINEA Z")


def test_quality_report_columns():
    q = D.quality_report(D.standardize(_raw(n_days=10)))
    assert q["total_consistente_pct"].iloc[0] == 100.0 and q["días"].iloc[0] == 10
