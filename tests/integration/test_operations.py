"""F8.7 — Operación diaria: auditoría, idempotencia, fallos controlados y monitoreo."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from metro_anomaly import operations as O
from metro_anomaly import service as S

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow
MODEL = ROOT / "artifacts" / "modelo.json"


@pytest.fixture(scope="module")
def hist():
    return S.load_history(ROOT / "data")


def test_process_logs_and_is_idempotent(hist, tmp_path):
    log = tmp_path / "d.jsonl"
    s1 = O.process(pd.date_range("2023-08-20", "2023-08-31"), hist, MODEL, log)
    assert s1["evaluados"] == 12 and s1["alertas"] == ["2023-08-28"]
    s2 = O.process(pd.date_range("2023-08-20", "2023-08-31"), hist, MODEL, log)
    assert s2["evaluados"] == 0 and s2["omitidos_ya_registrados"] == 12
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(recs) == 12 and all(len(r["artefacto_sha256"]) == 64 and r["datos_sha256"] for r in recs)
    r28 = next(r for r in recs if r["fecha"] == "2023-08-28")
    assert r28["categoria"] == "anomalia_estadistica" and r28["severidad"] == "extrema" and "18" in r28["horas_fuera_de_banda"]


def test_unavailable_day_reported_not_crash(hist, tmp_path):
    s = O.process(["2020-06-13", "2020-06-14"], hist, MODEL, tmp_path / "d.jsonl")
    assert s["evaluados"] == 1 and s["no_evaluables"][0]["fecha"] == "2020-06-13"


@pytest.mark.parametrize("content,msg", [(None, "No existe"), ("{no es json", "corrupto"), ('{"schema_version": 99}', "corrupto")])
def test_model_failures_controlled(hist, tmp_path, content, msg):
    p = tmp_path / "m.json"
    if content is not None:
        p.write_text(content)
    with pytest.raises(O.ModelError, match=msg):
        O.process(["2023-08-28"], hist, p, tmp_path / "d.jsonl")


def test_cli_exit_codes(tmp_path):
    base = [sys.executable, str(ROOT / "scripts/run_daily.py"), "--log", str(tmp_path / "d.jsonl")]
    ok = subprocess.run(base + ["--fecha", "2023-08-28", "--monitor"], capture_output=True, text=True)
    assert ok.returncode == 0 and '"alertas": ["2023-08-28"]' in ok.stdout
    bad = subprocess.run(base + ["--fecha", "2023-08-28", "--modelo", str(tmp_path / "x.json")], capture_output=True, text=True)
    assert bad.returncode == 3 and "error_modelo" in bad.stdout
    (tmp_path / "f.txt").write_text("x")
    inp = subprocess.run(base + ["--archivo", str(tmp_path / "f.txt")], capture_output=True, text=True)
    assert inp.returncode == 2 and "error_entrada" in inp.stdout


def test_monitor_flags_high_alert_rate(hist, tmp_path):
    log = tmp_path / "d.jsonl"
    O.process(pd.date_range("2023-12-01", "2024-01-31"), hist, MODEL, log)
    with open(log, "a") as fh:
        fh.write("línea dañada\n")
    m = O.monitor(log)
    assert m["registros"] == 62 and m["lineas_dañadas"] == 1
    assert m["por_mes"]["2023-12"]["dias"] == 31
