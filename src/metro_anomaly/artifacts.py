"""Serialización del modelo final en JSON (sin pickle: transparente y sin riesgo de ejecución de código).

Soporta los detectores de perfil (``perfil_robusto``/``perfil_contextual``), que son los elegibles para
producción según la selección en DEV. Guarda tablas de mediana y sigma por tipo de día, escalas de
agregación, umbral congelado y metadatos de procedencia (hash de los datos, commit, periodos, métricas).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import SCORED_HOURS
from .detectors import REGISTRY, ProfileDetector

SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def save_profile(det: ProfileDetector, threshold: float, path: str | Path, meta: dict) -> Path:
    if not isinstance(det, ProfileDetector):
        raise TypeError("save_profile solo admite detectores de perfil")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "detector": det.name, "agg": det.agg, "threshold": float(threshold),
        "params": {"window_days": det.window_days, "min_count": det.min_count},
        "hours": SCORED_HOURS,
        "median": {t: [float(x) for x in v.to_numpy()] for t, v in det.med_.items()},
        "sigma": {t: [float(x) for x in v.to_numpy()] for t, v in det.sig_.items()},
        "agg_stats": det.agg_stats_,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_profile(path: str | Path) -> tuple[ProfileDetector, float, dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Versión de artefacto no soportada: {doc.get('schema_version')}")
    if doc["hours"] != SCORED_HOURS:
        raise ValueError("Las horas del artefacto no coinciden con las del código")
    cls = REGISTRY[doc["detector"]]
    det = cls(agg=doc["agg"], **doc["params"])
    det.med_ = {t: pd.Series(v, index=SCORED_HOURS) for t, v in doc["median"].items()}
    det.sig_ = {t: pd.Series(v, index=SCORED_HOURS) for t, v in doc["sigma"].items()}
    det.agg_stats_ = doc["agg_stats"]
    return det, float(doc["threshold"]), doc
