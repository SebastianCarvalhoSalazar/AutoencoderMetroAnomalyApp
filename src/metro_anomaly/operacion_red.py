"""Operación diaria multilínea (fase 3): evaluación por lote de las 11 líneas, pronóstico del día siguiente,
registro de auditoría idempotente y monitoreo por línea.

Usa la misma lógica causal que los artefactos de `artifacts/plataforma/` (los modelos se ajustan con datos
anteriores al mes evaluado) y los umbrales y la selección de modelos congelados en `manifiesto.json`. Así, el
resultado operativo de un día coincide con el de la reproducción histórica (verificado en
tests/integration/test_operacion_red.py).

Registros (JSON Lines):
- decisiones: fecha, línea, categoría, score, umbral, alcance, horas fuera, nota de calendario, SHA-256 del
  manifiesto y de los datos de la línea en el día, versión del paquete, marca de tiempo.
- pronósticos: fecha objetivo, línea, hora, predicción, p10, p90, modelo, emitido_con_datos_hasta, SHA-256.
Reprocesar con el mismo manifiesto y los mismos datos no duplica registros.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import data as D
from . import plataforma as PL
from .artifacts import sha256_file
from .pronostico import forecast_day

ALERT_RATE_RANGE = (0.0, 0.10)  # por línea: por encima del 10 % en 90 días se advierte posible cambio de régimen
COVERAGE_RANGE = (0.70, 0.90)  # cobertura aceptable del intervalo nominal del 80 %
MIN_FORECAST_DAYS = 20


class ManifestError(RuntimeError):
    """Manifiesto de la plataforma ausente, corrupto o incompatible."""


def load_manifest_safe(path: str | Path) -> tuple[dict, str]:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"No existe el manifiesto de la plataforma: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ManifestError(f"Manifiesto inválido o corrupto ({type(e).__name__})") from None
    if doc.get("schema_version") != PL.SCHEMA_VERSION or "umbrales" not in doc or "seleccion_pronostico" not in doc:
        raise ManifestError("Manifiesto incompatible: versión de esquema o campos requeridos ausentes")
    return doc, sha256_file(p)


def parse_upload_red(content: bytes, filename: str) -> dict:
    """Archivo oficial (.xlsx/.csv) con una o varias líneas -> {línea: tabla diaria SCORED_HOURS}."""
    if not content:
        raise ValueError("El archivo está vacío.")
    suffix = Path(filename).suffix.lower()
    if suffix not in (".xlsx", ".csv"):
        raise ValueError("Formato no soportado: use .xlsx o .csv con el formato oficial de afluencia.")
    raw = pd.read_excel(io.BytesIO(content)) if suffix == ".xlsx" else pd.read_csv(io.BytesIO(content))
    std = D.standardize(raw, source=filename)
    out = {}
    for ln, g in std.groupby("line"):
        if ln in PL.MAIN_LINES:
            out[ln] = g.set_index("date").sort_index()[D.SCORED_HOURS]
    if not out:
        raise ValueError(f"El archivo no contiene líneas modeladas. Líneas presentes: {sorted(set(std['line']))}.")
    return out


def merge_red(red: PL.Red, new: dict | None) -> PL.Red:
    """Agrega o reemplaza (por fecha) los días nuevos de cada línea."""
    if not new:
        return red
    Ws = {}
    for ln, W in red.Ws.items():
        if ln in new:
            W = pd.concat([W[~W.index.isin(new[ln].index)], new[ln]]).sort_index()
        Ws[ln] = W
    return PL.Red(Ws, red.lineas, red.topologia, red.cierres)


def _row_hash(row: pd.Series) -> str:
    vals = ",".join("nan" if np.isnan(v) else f"{v:.0f}" for v in row.to_numpy(float))
    return hashlib.sha256(vals.encode()).hexdigest()[:16]


def _keys(log: Path, fields) -> set:
    if not log.exists():
        return set()
    out = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out.add(tuple(r[f] for f in fields))
        except Exception:  # noqa: BLE001 — una línea dañada no impide operar; monitor() la informa
            continue
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def process_red(dates, red: PL.Red, manifest_path: str | Path, log_path: str | Path) -> dict:
    """Evalúa ``dates`` en las 11 líneas y agrega los registros nuevos al log de decisiones."""
    doc, man_sha = load_manifest_safe(manifest_path)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    log = Path(log_path); log.parent.mkdir(parents=True, exist_ok=True)
    seen = _keys(log, ("fecha", "linea", "manifiesto_sha256", "datos_sha256"))
    summary = {"evaluados": 0, "omitidos_ya_registrados": 0, "sin_datos": [], "por_categoria": {}, "alertas": [],
               "ausencias_simultaneas": []}
    have = [d for d in dates if any(d in W.index and W.loc[d].notna().any() for W in red.Ws.values())]
    summary["sin_datos"] = [str(d.date()) for d in dates if d not in have]
    if not have:
        return summary
    horas, dias = PL.build_history(red, min(have), max(have))
    dias = dias[dias["fecha"].isin(have)]
    if dias.empty:
        return summary
    ev = PL.evaluate_days(red, horas, dias, doc["umbrales"])
    with open(log, "a", encoding="utf-8") as fh:
        for r in ev.itertuples(index=False):
            d, ln = pd.Timestamp(r.fecha), r.linea
            # reindex: una línea "sin registro" no tiene fila ese día (defecto D11 hallado por la simulación)
            key = (str(d.date()), ln, man_sha, _row_hash(red.Ws[ln].reindex([d])[D.SCORED_HOURS].iloc[0]))
            if key in seen:
                summary["omitidos_ya_registrados"] += 1
                continue
            rec = {"fecha": key[0], "linea": ln, "categoria": r.categoria,
                   "score": None if pd.isna(r.score) else round(float(r.score), 4),
                   "umbral": None if pd.isna(r.umbral) else round(float(r.umbral), 4),
                   "alcance": r.alcance or None, "horas_fuera": r.horas_fuera, "lineas_acompanan": r.lineas_acompanan,
                   "lineas_sentido_opuesto": r.lineas_sentido_opuesto, "calendario": r.nota_calendario or None,
                   "manifiesto_sha256": man_sha, "datos_sha256": key[3], "version_paquete": __version__, "procesado_utc": _now()}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(key)
            summary["evaluados"] += 1
            summary["por_categoria"][r.categoria] = summary["por_categoria"].get(r.categoria, 0) + 1
            if r.categoria.startswith("anomalia"):
                summary["alertas"].append(f"{key[0]} {ln}")
            elif r.categoria == "ausencia_simultanea_red":
                summary["ausencias_simultaneas"].append(f"{key[0]} {ln}")
            elif r.categoria == "linea_sin_registro":
                summary.setdefault("lineas_sin_registro", []).append(f"{key[0]} {ln}")
    return summary


def forecast_red(target, red: PL.Red, manifest_path: str | Path, log_path: str | Path) -> dict:
    """Emite el pronóstico de ``target`` (con datos < target) y lo registra una sola vez por manifiesto y fecha."""
    doc, man_sha = load_manifest_safe(manifest_path)
    target = pd.Timestamp(target).normalize()
    last = target - pd.Timedelta(days=1)
    data_sha = hashlib.sha256("".join(_row_hash(W.loc[:last].tail(1)[D.SCORED_HOURS].iloc[0]) if len(W.loc[:last]) else "-"
                                      for W in red.Ws.values()).encode()).hexdigest()[:16]
    log = Path(log_path); log.parent.mkdir(parents=True, exist_ok=True)
    if (str(target.date()), man_sha, data_sha) in _keys(log, ("fecha", "manifiesto_sha256", "datos_sha256")):
        return {"fecha": str(target.date()), "emitido": False, "motivo": "ya registrado"}
    F = forecast_day(red.Ws, target, doc["seleccion_pronostico"], doc.get("factor_conformal", {}), cierres=red.cierres)
    with open(log, "a", encoding="utf-8") as fh:
        for r in F.itertuples(index=False):
            fh.write(json.dumps({"fecha": str(target.date()), "linea": r.linea, "hora": int(r.hora), "pred": round(float(r.pred), 1),
                                 "p10": round(float(r.p10), 1), "p90": round(float(r.p90), 1), "modelo": r.modelo,
                                 "ratio_p95_gbm": round(float(r.ratio_p95_gbm), 4), "datos_hasta": str(last.date()),
                                 "manifiesto_sha256": man_sha, "datos_sha256": data_sha, "version_paquete": __version__,
                                 "procesado_utc": _now()}, ensure_ascii=False) + "\n")
    alta = sorted({r.linea for r in F.itertuples() if r.linea in PL.H6_LINES and r.ratio_p95_gbm > 1})
    return {"fecha": str(target.date()), "emitido": True, "filas": int(len(F)), "lineas_alta_demanda": alta}


def _read_jsonl(path: Path) -> tuple[pd.DataFrame, int]:
    rows, bad = [], 0
    if not path.exists():
        return pd.DataFrame(), 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            bad += 1
    return pd.DataFrame(rows), bad


def monitor_red(decisions_log: str | Path, forecasts_log: str | Path | None = None, red: PL.Red | None = None) -> dict:
    """Salud por línea: tasa de alertas (90 días), ausencias simultáneas, problemas de datos, versiones del manifiesto
    y, si hay pronósticos con dato observado, cobertura del intervalo del 80 % y WAPE por línea."""
    df, bad = _read_jsonl(Path(decisions_log))
    out = {"registros": int(len(df)), "lineas_dañadas": bad, "por_linea": {}, "advertencias": []}
    if len(df):
        df = df.drop_duplicates(["fecha", "linea"], keep="last").sort_values("fecha")
        df["alerta"] = df["categoria"].str.startswith("anomalia")
        scored = doc_lines_with_threshold(df)
        for ln, g in df.groupby("linea"):
            last90 = g.tail(90)
            rate = float(last90["alerta"].mean())
            out["por_linea"][ln] = {"dias": int(len(g)), "tasa_alertas_90d": round(rate, 4),
                                    "problemas_datos_90d": int((last90["categoria"] == "problema_de_datos").sum()),
                                    "ausencias_simultaneas_90d": int((last90["categoria"] == "ausencia_simultanea_red").sum()),
                                    "dias_sin_registro_90d": int((last90["categoria"] == "linea_sin_registro").sum()),
                                    "cierres_programados_90d": int((last90["categoria"] == "cierre_programado").sum())}
            if len(last90) >= 60 and rate > ALERT_RATE_RANGE[1]:
                out["advertencias"].append(f"{ln}: tasa de alertas de 90 días {rate:.1%} > {ALERT_RATE_RANGE[1]:.0%} "
                                           "(posible cambio de régimen; revisar y considerar reconstruir los artefactos).")
            if len(g) >= 180 and not g.tail(180)["alerta"].any() and ln in scored:
                out["advertencias"].append(f"{ln}: sin alertas en 180 días; verificar el flujo de datos.")
        if df["manifiesto_sha256"].nunique() > 1:
            out["advertencias"].append(f"Se usaron {df['manifiesto_sha256'].nunique()} versiones del manifiesto en el periodo.")
    if forecasts_log is not None and red is not None:
        out["pronostico"] = forecast_coverage(forecasts_log, red)
        for ln, v in out["pronostico"].get("por_linea", {}).items():
            if v["dias"] >= MIN_FORECAST_DAYS and not (COVERAGE_RANGE[0] <= v["cobertura_80"] <= COVERAGE_RANGE[1]):
                out["advertencias"].append(f"{ln}: cobertura del intervalo del 80 % = {v['cobertura_80']:.2f} fuera de "
                                           f"[{COVERAGE_RANGE[0]}, {COVERAGE_RANGE[1]}] en {v['dias']} días; recalibrar.")
    return out


def doc_lines_with_threshold(df: pd.DataFrame) -> set:
    """Líneas con score (umbral definido) en el log: solo en ellas tiene sentido esperar alertas."""
    return set(df.loc[df["umbral"].notna(), "linea"]) if "umbral" in df else set()


def forecast_coverage(forecasts_log: str | Path, red: PL.Red) -> dict:
    """Compara los pronósticos registrados con el dato observado (cuando ya existe)."""
    F, bad = _read_jsonl(Path(forecasts_log))
    if F.empty:
        return {"filas": 0, "lineas_dañadas": bad}
    F = F.drop_duplicates(["fecha", "linea", "hora"], keep="last")
    F["fecha"] = pd.to_datetime(F["fecha"])
    obs = []
    for r in F.itertuples(index=False):
        W = red.Ws.get(r.linea)
        col = f"h{int(r.hora):02d}"
        obs.append(W.at[r.fecha, col] if W is not None and r.fecha in W.index else np.nan)
    F["observado"] = obs
    F = F.dropna(subset=["observado"])
    res = {"filas_con_dato": int(len(F)), "lineas_dañadas": bad, "por_linea": {}}
    for ln, g in F.groupby("linea"):
        res["por_linea"][ln] = {"dias": int(g["fecha"].nunique()),
                                "cobertura_80": round(float(((g["observado"] >= g["p10"]) & (g["observado"] <= g["p90"])).mean()), 3),
                                "wape": round(float((g["observado"] - g["pred"]).abs().sum() / max(g["observado"].sum(), 1e-9)), 4)}
    return res
