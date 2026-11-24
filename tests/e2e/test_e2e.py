"""F5.3 — E2E con navegador real (Playwright para Node). Se omite si no hay node/playwright."""
import datetime as dt
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metro_anomaly import data as D

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


def _files(tmp):
    df, _ = D.load_all(ROOT / "data")
    w = D.line_series(df, "LINEA A").loc["2024-05-01":"2024-06-30", D.HOUR_COLS].copy()
    w.loc["2024-06-12", ["h17", "h18"]] *= 0.1
    out = pd.DataFrame({"Día": w.index.strftime("%d/%m/%Y"), "Línea de Servicio": "LÍNEA A"})
    for h, c in zip(D.HOURS, D.HOUR_COLS):
        out[dt.time(h)] = w[c].to_numpy()
    good = tmp / "valido.xlsx"; out.to_excel(good, index=False)
    bad = tmp / "invalido.csv"; bad.write_text("x,y\n1,2\n")
    return good, bad


def test_app_e2e(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node no disponible")
    env = dict(os.environ, NODE_PATH=subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip())
    if subprocess.run([node, "-e", "require('playwright')"], env=env).returncode != 0:
        pytest.skip("playwright (node) no disponible")
    good, bad = _files(tmp_path)
    shots = Path(os.environ.get("E2E_SHOTS", tmp_path)); shots.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                            "--server.port", str(port), "--browser.gatherUsageStats", "false"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close(); break
            except OSError:
                time.sleep(1)
        r = subprocess.run([node, str(ROOT / "tests/e2e/app.e2e.js"), f"http://127.0.0.1:{port}", str(good), str(bad), str(shots)],
                           env=env, capture_output=True, text=True, timeout=600)
        print(r.stdout, r.stderr)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "E2E OK" in r.stdout
    finally:
        srv.terminate(); srv.wait(timeout=30)


def test_red_multilinea_e2e(tmp_path):
    """F10.6 — Página multilínea: capturas, reproducción por hora, verificación visual-datos y desbordamiento."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node no disponible")
    if not (ROOT / "artifacts" / "plataforma" / "manifiesto.json").exists():
        pytest.skip("faltan los artefactos de la plataforma")
    env = dict(os.environ, NODE_PATH=subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip())
    if subprocess.run([node, "-e", "require('playwright')"], env=env).returncode != 0:
        pytest.skip("playwright (node) no disponible")
    out = Path(os.environ.get("E2E_SHOTS", tmp_path)) / "red"
    port = _free_port()
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                            "--server.port", str(port), "--browser.gatherUsageStats", "false"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close(); break
            except OSError:
                time.sleep(1)
        r = subprocess.run([node, str(ROOT / "tests/e2e/red_multilinea.js"), f"http://127.0.0.1:{port}", str(out)],
                           env=env, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout + r.stderr
        c = subprocess.run([sys.executable, str(ROOT / "tests/e2e/check_red_multilinea.py"), str(out)], capture_output=True, text=True)
        assert c.returncode == 0, c.stdout + c.stderr
    finally:
        srv.terminate(); srv.wait(timeout=30)
