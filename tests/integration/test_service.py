"""F5.1 — Servicio de la app: validación, puntuación y explicación (con el artefacto y datos reales)."""
import datetime as dt
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metro_anomaly import data as D
from metro_anomaly import service as S

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def env():
    hist = S.load_history(ROOT / "data")
    det, thr, doc = S.load_model(ROOT / "artifacts" / "modelo.json")
    return hist, det, thr


def _official_bytes(hist, dates, fmt="xlsx", line="LÍNEA A", mutate=None):
    rows = hist.loc[dates, D.HOUR_COLS].copy()
    if mutate:
        rows = mutate(rows)
    df = pd.DataFrame({"Día": rows.index.strftime("%d/%m/%Y"), "Línea de Servicio": line})
    for h, c in zip(D.HOURS, D.HOUR_COLS):
        df[dt.time(h) if fmt == "xlsx" else f"{h:02d}:00:00"] = rows[c].to_numpy()
    df[D.TOTAL_COL] = rows.sum(axis=1).to_numpy()
    buf = io.BytesIO()
    (df.to_excel(buf, index=False) if fmt == "xlsx" else df.to_csv(buf, index=False))
    return buf.getvalue()


def test_known_incident_day_is_alert_and_explained(env):
    hist, det, thr = env
    r = S.score_days(det, thr, hist, ["2023-08-28"])[0]
    assert r.alert and {"h18", "h19"} <= set(r.hour_flags)
    assert (r.observed["h19"] < r.band_low["h19"])


def test_election_day_annotated(env):
    hist, det, thr = env
    r = S.score_days(det, thr, hist, ["2023-10-29"])[0]
    assert r.alert and "Elecciones" in r.note


def test_normal_day_not_alert(env):
    hist, det, thr = env
    r = S.score_days(det, thr, hist, ["2024-03-06"])[0]
    assert not r.alert and r.hour_flags == [] and abs(r.deviation_pct) < 15


def test_recent_normal_week_has_no_alerts(env):
    """Semana ordinaria (dentro de la ventana de entrenamiento del artefacto): scores finitos y sin alertas."""
    hist, det, thr = env
    rs = S.score_days(det, thr, hist, pd.date_range("2024-06-03", "2024-06-07"))
    assert all(np.isfinite(r.score) for r in rs) and sum(r.alert for r in rs) == 0


@pytest.mark.parametrize("fmt", ["xlsx", "csv"])
def test_upload_roundtrip(env, fmt):
    hist, det, thr = env
    b = _official_bytes(hist, pd.date_range("2024-05-01", "2024-06-30"), fmt)
    up = S.parse_upload(b, f"x.{fmt}")
    assert len(up) == 61
    ctx = S.merge_context(hist, up)
    rs = S.score_days(det, thr, ctx, up.index[-7:])
    assert len(rs) == 7


def test_upload_future_days_with_own_context(env):
    """Días posteriores al histórico: el archivo aporta su propio contexto (fechas desplazadas un año)."""
    hist, det, thr = env
    b = _official_bytes(hist, pd.date_range("2023-07-03", "2023-09-30"), mutate=lambda r: r.set_axis(r.index + pd.Timedelta(days=364)))
    up = S.parse_upload(b, "futuro.xlsx")
    assert up.index.min() > hist.index.max()
    rs = S.score_days(det, thr, S.merge_context(hist, up), up.index[-20:])
    assert len(rs) == 20 and all(np.isfinite(r.score) for r in rs)


def test_injected_interruption_detected_via_upload(env):
    hist, det, thr = env

    def cut(rows):
        rows.loc["2024-06-12", ["h17", "h18"]] *= 0.1
        return rows
    b = _official_bytes(hist, pd.date_range("2024-05-01", "2024-06-30"), mutate=cut)
    up = S.parse_upload(b, "x.xlsx")
    r = S.score_days(det, thr, S.merge_context(hist, up), ["2024-06-12"])[0]
    assert r.alert and set(r.hour_flags) >= {"h17", "h18"}


def test_missing_hour_reported(env):
    hist, det, thr = env

    def gap(rows):
        rows.loc["2024-06-12", "h10"] = np.nan
        return rows
    up = S.parse_upload(_official_bytes(hist, pd.date_range("2024-05-01", "2024-06-30"), mutate=gap), "x.xlsx")
    r = S.score_days(det, thr, S.merge_context(hist, up), ["2024-06-12"])[0]
    assert r.missing_hours == ["h10"] and np.isfinite(r.score)


@pytest.mark.parametrize("content,name,msg", [
    (b"", "a.xlsx", "vacío"),
    (b"hola", "a.txt", "Formato no soportado"),
    (b"\x00\x01garbage", "a.xlsx", "No se pudo leer"),
    (b"a,b\n1,2\n", "a.csv", "faltan columnas"),
])
def test_bad_uploads(content, name, msg):
    with pytest.raises(S.InputError, match=msg):
        S.parse_upload(content, name)


def test_other_line_rejected(env):
    hist, _, _ = env
    with pytest.raises(S.InputError, match="LINEA A"):
        S.parse_upload(_official_bytes(hist, pd.date_range("2024-06-01", "2024-06-05"), line="LÍNEA B"), "b.xlsx")


def test_negative_values_rejected(env):
    hist, _, _ = env
    b = _official_bytes(hist, pd.date_range("2024-06-01", "2024-06-05"), mutate=lambda r: r.assign(h10=-5.0))
    with pytest.raises(S.InputError, match="negativos"):
        S.parse_upload(b, "x.xlsx")


def test_insufficient_history(env):
    hist, det, thr = env
    b = _official_bytes(hist, pd.date_range("2023-07-03", "2023-07-05"), mutate=lambda r: r.set_axis(r.index + pd.Timedelta(days=728)))
    up = S.parse_upload(b, "x.xlsx")
    with pytest.raises(S.InputError, match="Historia insuficiente"):
        S.score_days(det, thr, S.merge_context(hist, up), up.index)


def test_date_without_data(env):
    hist, det, thr = env
    with pytest.raises(S.InputError, match="No hay datos"):
        S.score_days(det, thr, hist, ["2020-06-13"])


def test_summary_table(env):
    hist, det, thr = env
    t = S.summary_table(S.score_days(det, thr, hist, ["2023-08-28", "2024-03-06"]))
    assert t["Estado"].tolist() == ["ANOMALÍA", "Normal"]
