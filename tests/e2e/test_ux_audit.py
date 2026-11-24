"""F9 — Auditoría funcional/visual/UX completa con navegador real + consistencia UI↔backend.

Requiere node + playwright; axe-core es opcional (variable AXE_PATH -> axe.min.js; p. ej.
`npm install axe-core@4.10.2` y AXE_PATH=node_modules/axe-core/axe.min.js). Capturas en UX_SHOTS (o tmp).
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_ux_audit(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node no disponible")
    env = dict(os.environ, NODE_PATH=subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip())
    if subprocess.run([node, "-e", "require('playwright')"], env=env).returncode != 0:
        pytest.skip("playwright (node) no disponible")
    fx = tmp_path / "fixtures"
    subprocess.run([sys.executable, str(ROOT / "tests/e2e/make_ux_fixtures.py"), str(fx)], check=True, capture_output=True)
    shots = Path(os.environ.get("UX_SHOTS", tmp_path / "shots"))
    out = tmp_path / "audit.json"
    port = _port()
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                            "--server.port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close(); break
            except OSError:
                time.sleep(1)
        args = [node, str(ROOT / "tests/e2e/ux_audit.js"), f"http://127.0.0.1:{port}", str(fx), str(shots), str(out)]
        if os.environ.get("AXE_PATH"):
            args.append(os.environ["AXE_PATH"])
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=2400)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    finally:
        srv.terminate(); srv.wait(timeout=30)
    a = json.loads(out.read_text())
    steps = a["steps"]
    assert all(s["dateInput"]["applied"] for s in steps if "dateInput" in s), "alguna fecha no se aplicó"
    assert sum(len(s["consoleErrors"]) + len(s["pageErrors"]) for s in steps) == 0          # UX1
    assert sum(s["exceptions"] for s in steps) == 0 and sum(len(s["failedRequests"]) for s in steps) == 0
    for k, v in a["responsive"].items():                                                    # UX5
        assert v["docScroll"] <= v["docClient"], k
    framework = {"aria-allowed-attr", "button-name", "summary-name", "nested-interactive"}  # internos de Streamlit/Vega
    for view, viol in a.get("axe", {}).items():                                             # UX4
        assert not [v for v in viol if v["impact"] in ("critical", "serious") and v["id"] not in framework], (view, viol)
    chk = subprocess.run([sys.executable, str(ROOT / "tests/e2e/check_ux_audit.py"), str(out), str(fx), str(tmp_path / "c.json")],
                         capture_output=True, text=True)
    assert chk.returncode == 0, chk.stdout[-2000:]                                          # UX3
