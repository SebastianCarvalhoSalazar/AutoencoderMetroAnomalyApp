"""F6.1 / D9 — Ejecuta scripts/reproduce.py dos veces en copias aisladas del repositorio y compara los hashes
de las salidas (sin TensorFlow). Escribe reports/determinismo.md en el repositorio original."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ["configs/seleccion.yaml", "reports/seleccion_dev.md", "reports/holdout.md", "reports/holdout_metricas.csv",
           "reports/externo.md", "reports/calidad_datos.md", "reports/etiquetas.md", "artifacts/modelo.json"]
VOLATILE = {"created_utc", "git_commit_build"}


def digest(p: Path) -> str:
    b = p.read_bytes()
    if p.suffix == ".json":
        d = {k: v for k, v in json.loads(b).items() if k not in VOLATILE}
        b = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(b).hexdigest()


def one_run(tag: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"det_{tag}_"))
    for d in ("src", "scripts", "data"):
        shutil.copytree(ROOT / d, tmp / d)
    env = dict(os.environ, PYTHONPATH=str(tmp / "src"))
    r = subprocess.run([sys.executable, str(tmp / "scripts" / "reproduce.py"), "--workers", "2"], cwd=tmp, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"reproduce falló en {tag}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return {o: digest(tmp / o) for o in OUTPUTS}


def main():
    a, b = one_run("A"), one_run("B")
    rows = [f"| `{o}` | `{a[o][:16]}` | `{b[o][:16]}` | {'✔' if a[o] == b[o] else '✘'} |" for o in OUTPUTS]
    ok = all(a[o] == b[o] for o in OUTPUTS)
    md = ["# Determinismo (D9) — generado por `scripts/check_determinism.py`", "",
          "Dos ejecuciones completas de `scripts/reproduce.py` (sin TensorFlow) en copias aisladas del repositorio. "
          "En el artefacto se ignoran los campos volátiles `created_utc` y `git_commit_build`.", "",
          "| Salida | SHA-256 ejecución A | SHA-256 ejecución B | Idéntico |", "|---|---|---|---|", *rows, "",
          f"**Resultado: {'DETERMINISTA ✔' if ok else 'NO DETERMINISTA ✘'}**"]
    (ROOT / "reports" / "determinismo.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
