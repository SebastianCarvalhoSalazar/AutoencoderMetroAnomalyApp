"""Fase 3 — Operación diaria multilínea: coherencia con los artefactos, idempotencia, pronóstico, monitoreo y
fallos controlados."""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metro_anomaly import data as D
from metro_anomaly import operacion_red as OR
from metro_anomaly import plataforma as PL

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "artifacts" / "plataforma" / "manifiesto.json"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not MAN.exists(), reason="faltan artefactos")]


@pytest.fixture(scope="module")
def red():
    return PL.load_network(ROOT / "data")


@pytest.fixture(scope="module")
def ran(red, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ops")
    log = tmp / "d.jsonl"
    s1 = OR.process_red(pd.date_range("2023-08-26", "2023-08-28"), red, MAN, log)
    return tmp, log, s1


def test_operation_matches_platform_artifacts(ran):
    """La operación diaria reproduce exactamente la categoría de la reproducción histórica (33 línea-días)."""
    _, log, s1 = ran
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert s1["evaluados"] == len(recs) == 33
    art = pd.read_csv(ROOT / "artifacts" / "plataforma" / "dias.csv.gz", parse_dates=["fecha"]).set_index(["fecha", "linea"])
    for r in recs:
        assert art.loc[(pd.Timestamp(r["fecha"]), r["linea"]), "categoria"] == r["categoria"]
    a28 = next(r for r in recs if r["fecha"] == "2023-08-28" and r["linea"] == "LINEA A")
    assert a28["categoria"] == "anomalia_multilinea" and len(a28["manifiesto_sha256"]) == 64 and a28["datos_sha256"]
    assert {"2023-08-28 LINEA B", "2023-08-28 LINEA J", "2023-08-28 LINEA K"} <= set(s1["ausencias_simultaneas"])


def test_idempotent(red, ran):
    _, log, _ = ran
    s2 = OR.process_red(pd.date_range("2023-08-26", "2023-08-28"), red, MAN, log)
    assert s2["evaluados"] == 0 and s2["omitidos_ya_registrados"] == 33


def test_early_closure_annotated(red, tmp_path):
    s = OR.process_red(["2023-12-24"], red, MAN, tmp_path / "d.jsonl")
    recs = [json.loads(x) for x in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()]
    assert s["evaluados"] == 11
    assert all("Cierre anticipado" in (r["calendario"] or "") for r in recs)


def test_day_without_data(red, tmp_path):
    s = OR.process_red(["2030-01-01"], red, MAN, tmp_path / "d.jsonl")
    assert s["evaluados"] == 0 and s["sin_datos"] == ["2030-01-01"]


@pytest.mark.parametrize("content,msg", [(None, "No existe"), ("{no es json", "corrupto"), ('{"schema_version": 99}', "incompatible")])
def test_manifest_failures_controlled(red, tmp_path, content, msg):
    p = tmp_path / "m.json"
    if content is not None:
        p.write_text(content)
    with pytest.raises(OR.ManifestError, match=msg):
        OR.process_red(["2023-08-28"], red, p, tmp_path / "d.jsonl")


@pytest.fixture(scope="module")
def forecast(red, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fc")
    flog = tmp / "p.jsonl"
    out = OR.forecast_red("2023-08-29", red, MAN, flog)
    return flog, out


def test_forecast_emitted_once_and_causal(red, forecast):
    flog, out = forecast
    assert out["emitido"] and out["filas"] == 11 * 19
    recs = [json.loads(x) for x in flog.read_text(encoding="utf-8").splitlines()]
    assert all(r["datos_hasta"] == "2023-08-28" and r["p10"] <= r["pred"] <= r["p90"] for r in recs)
    again = OR.forecast_red("2023-08-29", red, MAN, flog)
    assert again["emitido"] is False


def test_monitor_reports_forecast_coverage(red, ran, forecast):
    _, log, _ = ran
    flog, _ = forecast
    m = OR.monitor_red(log, flog, red)
    assert m["registros"] == 33 and set(m["por_linea"]) == set(PL.MAIN_LINES)
    cov = m["pronostico"]["por_linea"]
    assert set(cov) == set(PL.MAIN_LINES) and all(0 <= v["cobertura_80"] <= 1 and v["wape"] > 0 for v in cov.values())
    assert not any("cobertura" in w for w in m["advertencias"])  # 1 día < mínimo de días para advertir


def test_monitor_flags_high_alert_rate(tmp_path):
    rows = [{"fecha": str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date()), "linea": "LINEA A",
             "categoria": "anomalia_local" if i % 5 == 0 else "sin_anomalia", "umbral": 9.4, "manifiesto_sha256": "x"}
            for i in range(90)]
    log = tmp_path / "d.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n{dañada", encoding="utf-8")
    m = OR.monitor_red(log)
    assert m["lineas_dañadas"] == 1 and m["por_linea"]["LINEA A"]["tasa_alertas_90d"] == 0.2
    assert any("LINEA A" in w and "tasa de alertas" in w for w in m["advertencias"])


def test_upload_multiline_file_and_merge(red, tmp_path):
    """Un archivo oficial con varias líneas se lee, se fusiona y se evalúa; sus valores reemplazan a los históricos."""
    df, _ = D.load_all(ROOT / "data")
    day = df[df["date"] == pd.Timestamp("2024-06-12")].copy()
    out = pd.DataFrame({"Día": day["date"].dt.strftime("%d/%m/%Y"), "Línea de Servicio": day["line"].str.replace("LINEA", "LÍNEA")})
    for h, c in zip(D.HOURS, D.HOUR_COLS):
        v = day[c].to_numpy().copy()
        if h in (17, 18):
            v = np.where(day["line"].to_numpy() == "LINEA B", v * 0.1, v)  # caída fuerte solo en la Línea B
        out[dt.time(h)] = v
    p = tmp_path / "multi.xlsx"; out.to_excel(p, index=False)
    new = OR.parse_upload_red(p.read_bytes(), p.name)
    assert {"LINEA A", "LINEA B", "LINEA 1"} <= set(new)
    red2 = OR.merge_red(red, new)
    assert red2.Ws["LINEA B"].at[pd.Timestamp("2024-06-12"), "h17"] == pytest.approx(red.Ws["LINEA B"].at[pd.Timestamp("2024-06-12"), "h17"] * 0.1, rel=1e-6)
    s = OR.process_red(["2024-06-12"], red2, MAN, tmp_path / "d.jsonl")
    assert "2024-06-12 LINEA B" in s["alertas"]
    recs = {json.loads(x)["linea"]: json.loads(x) for x in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()}
    assert recs["LINEA B"]["alcance"] == "local" and "17" in recs["LINEA B"]["horas_fuera"]


def test_cli_red_exit_codes(tmp_path):
    base = [sys.executable, str(ROOT / "scripts/run_daily.py"), "--red", "--log-red", str(tmp_path / "d.jsonl")]
    ok = subprocess.run(base + ["--fecha", "2023-10-29", "--monitor"], capture_output=True, text=True)
    assert ok.returncode == 0 and '"efecto_de_medicion": 8' in ok.stdout
    bad = subprocess.run(base + ["--fecha", "2023-10-29", "--manifiesto", str(tmp_path / "x.json")], capture_output=True, text=True)
    assert bad.returncode == 3 and "error_modelo" in bad.stdout
    (tmp_path / "f.txt").write_text("x")
    inp = subprocess.run(base + ["--archivo", str(tmp_path / "f.txt")], capture_output=True, text=True)
    assert inp.returncode == 2 and "error_entrada" in inp.stdout


def test_line_without_records_is_logged(red, tmp_path):
    """Defecto D11: un día en que una línea habitual no tiene ninguna fila debe registrarse, no romper el lote."""
    s = OR.process_red(["2024-06-27"], red, MAN, tmp_path / "d.jsonl")
    assert "2024-06-27 LINEA K" in s.get("lineas_sin_registro", [])
    recs = {json.loads(x)["linea"]: json.loads(x) for x in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()}
    assert recs["LINEA K"]["categoria"] == "linea_sin_registro" and recs["LINEA K"]["datos_sha256"]
    s2 = OR.process_red(["2024-06-12"], red, MAN, tmp_path / "d2.jsonl")
    assert s2["por_categoria"].get("cierre_programado") == 1  # Línea P en mantenimiento anunciado
