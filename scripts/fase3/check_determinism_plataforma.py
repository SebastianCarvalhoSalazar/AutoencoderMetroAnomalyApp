"""F10 — Determinismo de los artefactos de la plataforma: ejecuta scripts/fase3/build_plataforma.py dos veces en
copias aisladas del repositorio (sin .git) y compara los SHA-256 de las salidas. Escribe
reports/fase3/determinismo_plataforma.md en el repositorio original.

En el manifiesto se ignoran los campos volátiles (`creado`, `git_commit_build`, `segundos_construccion`).
Uso: python scripts/fase3/check_determinism_plataforma.py   (~25 min: dos construcciones completas)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ["dias.csv.gz", "horas.csv.gz", "pronostico_holdout.csv.gz", "pronostico_siguiente.csv", "manifiesto.json"]
VOLATILE = {"creado", "git_commit_build", "segundos_construccion"}
INPUTS = ["src", "scripts", "data"]


def digest(p: Path) -> str:
    if p.suffix == ".json":
        d = {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k not in VOLATILE}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    if p.suffix == ".gz":  # el encabezado gzip incluye la fecha de escritura: se compara el contenido descomprimido
        import gzip
        return hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest()


def one_run(tag: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"detplat_{tag}_"))
    for d in INPUTS:
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / "reports" / "fase3").mkdir(parents=True)
    shutil.copy(ROOT / "reports" / "fase3" / "pronostico_oos_holdout.csv.gz", tmp / "reports" / "fase3")
    shutil.copy(ROOT / "reports" / "fase3" / "pronostico_holdout.json", tmp / "reports" / "fase3")
    env = dict(os.environ, PYTHONPATH=str(tmp / "src"))
    r = subprocess.run([sys.executable, "-W", "ignore", str(tmp / "scripts" / "fase3" / "build_plataforma.py")], cwd=tmp, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"build_plataforma falló en {tag}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    out = {o: digest(tmp / "artifacts" / "plataforma" / o) for o in OUTPUTS}
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    a, b = one_run("A"), one_run("B")
    repo = {o: digest(ROOT / "artifacts" / "plataforma" / o) for o in OUTPUTS}
    rows = [f"| `{o}` | `{a[o][:16]}` | `{b[o][:16]}` | {'✔' if a[o] == b[o] else '✘'} | {'✔' if a[o] == repo[o] else '✘'} |"
            for o in OUTPUTS]
    ok = all(a[o] == b[o] for o in OUTPUTS)
    same_repo = all(a[o] == repo[o] for o in OUTPUTS)
    md = ["# Determinismo de los artefactos de la plataforma (generado por `scripts/fase3/check_determinism_plataforma.py`)", "",
          "Dos construcciones completas de `scripts/fase3/build_plataforma.py` en copias aisladas (sin `.git`). Los `.csv.gz` se "
          "comparan descomprimidos (el encabezado gzip lleva la fecha) y en el manifiesto se ignoran "
          "`creado`, `git_commit_build` y `segundos_construccion`.", "",
          "| Salida | SHA-256 A | SHA-256 B | A = B | A = versionado |", "|---|---|---|---|---|", *rows, "",
          f"**Resultado: {'DETERMINISTA ✔' if ok else 'NO DETERMINISTA ✘'}** · "
          f"{'coincide con los artefactos versionados ✔' if same_repo else 'difiere de los artefactos versionados ✘'}"]
    (ROOT / "reports" / "fase3" / "determinismo_plataforma.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    sys.exit(0 if ok and same_repo else 1)


if __name__ == "__main__":
    main()
