"""Operación diaria (F8.7): procesamiento por lote con registro de auditoría, idempotencia y monitoreo.

Cada decisión se registra en JSON Lines con: fecha evaluada, categoría, score, umbral, severidad, horas fuera
de banda, banderas de calidad, hash SHA-256 del artefacto y de los datos del día, versión del paquete y
marca de tiempo. Reprocesar el mismo día con el mismo artefacto y los mismos datos no duplica registros.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import data as D
from . import service as S
from .artifacts import sha256_file

EXIT_OK, EXIT_INPUT, EXIT_MODEL = 0, 2, 3
ALERT_RATE_RANGE = (0.02, 0.10)  # rango aceptable de la tasa de alertas (criterio D1)


class ModelError(RuntimeError):
    """Artefacto ausente, corrupto o incompatible."""


def load_model_safe(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise ModelError(f"No existe el artefacto del modelo: {p}")
    try:
        det, thr, doc = S.load_model(p)
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"Artefacto inválido o corrupto ({type(e).__name__}): {e}") from None
    return det, thr, doc, sha256_file(p)


def _row_hash(row: pd.Series) -> str:
    vals = ",".join("nan" if np.isnan(v) else f"{v:.0f}" for v in row.to_numpy(float))
    return hashlib.sha256(vals.encode()).hexdigest()[:16]


def _existing(log: Path) -> set:
    if not log.exists():
        return set()
    keys = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            keys.add((r["fecha"], r["artefacto_sha256"], r["datos_sha256"]))
        except Exception:  # noqa: BLE001 — una línea dañada no debe impedir operar; se informa en monitor()
            continue
    return keys


def process(dates, history: pd.DataFrame, model_path: str | Path, log_path: str | Path,
            new_data: pd.DataFrame | None = None) -> dict:
    """Evalúa ``dates`` y agrega los registros nuevos al log. Devuelve un resumen."""
    det, thr, doc, art_sha = load_model_safe(model_path)
    ref = S.reference_scores(doc)
    ctx = S.merge_context(history, new_data)
    log = Path(log_path); log.parent.mkdir(parents=True, exist_ok=True)
    seen = _existing(log)
    summary = {"evaluados": 0, "omitidos_ya_registrados": 0, "no_evaluables": [], "alertas": [], "problemas_de_datos": []}
    with open(log, "a", encoding="utf-8") as fh:
        for d in pd.DatetimeIndex(pd.to_datetime(dates)):
            try:
                r = S.score_days(det, thr, ctx, [d], ref)[0]
            except S.InputError as e:
                summary["no_evaluables"].append({"fecha": str(d.date()), "motivo": str(e)})
                continue
            key = (str(d.date()), art_sha, _row_hash(ctx.loc[d, D.SCORED_HOURS]))
            if key in seen:
                summary["omitidos_ya_registrados"] += 1
                continue
            rec = {"fecha": key[0], "linea": S.MODEL_LINE, "categoria": r.category, "estado": r.status,
                   "score": round(r.score, 4), "umbral": round(thr, 4), "severidad": r.severity,
                   "frecuencia_historica": None if r.exceedance is None else round(r.exceedance, 4),
                   "horas_fuera_de_banda": [h[1:] for h in r.hour_flags], "calidad_datos": r.dq_flags,
                   "calendario": r.note, "artefacto_sha256": art_sha, "datos_sha256": key[2],
                   "version_paquete": __version__, "procesado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(key)
            summary["evaluados"] += 1
            if r.category == "anomalia_estadistica":
                summary["alertas"].append(key[0])
            elif r.category in ("posible_problema_de_datos", "efecto_de_medicion"):
                summary["problemas_de_datos"].append(key[0])
    return summary


def monitor(log_path: str | Path) -> dict:
    """Indicadores de salud: tasa de alertas por mes frente al rango esperado, problemas de datos, deriva."""
    rows, bad = [], 0
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            bad += 1
    if not rows:
        return {"registros": 0, "lineas_dañadas": bad}
    df = pd.DataFrame(rows).drop_duplicates(["fecha"], keep="last")
    df["mes"] = pd.to_datetime(df["fecha"]).dt.to_period("M").astype(str)
    df["alerta"] = df["categoria"] == "anomalia_estadistica"
    by = df.groupby("mes").agg(dias=("fecha", "size"), tasa_alertas=("alerta", "mean"),
                               problemas_datos=("categoria", lambda c: int((c == "posible_problema_de_datos").sum())))
    srt = df.sort_values("fecha")
    last90, last180 = srt.tail(90), srt.tail(180)
    rate90 = float(last90["alerta"].mean())
    warnings = []
    if rate90 > ALERT_RATE_RANGE[1]:
        warnings.append(f"Tasa de alertas de los últimos 90 días ({rate90:.1%}) por encima del {ALERT_RATE_RANGE[1]:.0%}: "
                        "posible cambio de régimen o degradación; revisar y considerar reentrenar o recalibrar.")
    # Con una tasa de ~2 %, P(0 alertas en 90 días) ≈ 13 % (aviso espurio frecuente); en 180 días ≈ 2 %.
    if len(last180) >= 180 and not last180["alerta"].any():
        warnings.append("Sin alertas en los últimos 180 días (probabilidad ≈ 2 % si el sistema funciona con su tasa "
                        "esperada): verificar el flujo de datos y el modelo.")
    if df["artefacto_sha256"].nunique() > 1:
        warnings.append(f"Se usaron {df['artefacto_sha256'].nunique()} versiones del artefacto en el periodo registrado.")
    return {"registros": int(len(df)), "lineas_dañadas": bad, "tasa_alertas_90d": round(rate90, 4),
            "por_mes": by.round(4).to_dict(orient="index"), "advertencias": warnings}
