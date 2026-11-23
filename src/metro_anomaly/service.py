"""Servicio de detección para la aplicación (sin dependencias de UI).

Flujo: ``load_history`` + ``load_model`` → ``parse_upload`` (opcional) → ``score_days`` → resultados explicados.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .artifacts import load_profile
from .calendar import day_type_ext, event_note
from .detectors import LEVEL_FALLBACK_DAYS, LEVEL_MIN_DAYS, level_series

MODEL_LINE = "LINEA A"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class InputError(ValueError):
    """Error de entrada mostrado al usuario (mensaje en español, sin traza)."""


@dataclass
class DayResult:
    date: pd.Timestamp
    score: float
    threshold: float
    alert: bool
    day_type: str
    note: str | None
    observed: pd.Series
    expected: pd.Series
    band_low: pd.Series
    band_high: pd.Series
    z: pd.Series
    hour_flags: list[str]
    deviation_pct: float
    missing_hours: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "ANOMALÍA" if self.alert else "Normal"


def load_history(data_dir: str | Path) -> pd.DataFrame:
    df, _ = D.load_all(data_dir)
    return D.line_series(df, MODEL_LINE)


def load_model(path: str | Path):
    return load_profile(path)


def parse_upload(content: bytes, filename: str, line: str = MODEL_LINE) -> pd.DataFrame:
    """Lee un .xlsx/.csv con el formato oficial y devuelve la tabla diaria de la línea indicada."""
    if not content:
        raise InputError("El archivo está vacío.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise InputError("El archivo supera el tamaño máximo permitido (20 MB).")
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".xlsx":
            raw = pd.read_excel(io.BytesIO(content))
        elif suffix == ".csv":
            raw = pd.read_csv(io.BytesIO(content))
        else:
            raise InputError("Formato no soportado: use .xlsx o .csv con el formato oficial de afluencia.")
    except InputError:
        raise
    except Exception as e:
        raise InputError(f"No se pudo leer el archivo: {type(e).__name__}.") from None
    try:
        std = D.standardize(raw, source=filename)
    except D.DataValidationError as e:
        raise InputError(str(e)) from None
    line = D.normalize_line(line)
    if line not in set(std["line"]):
        raise InputError(f"El archivo no contiene la línea {line}. Líneas presentes: {sorted(set(std['line']))}.")
    if line != MODEL_LINE:
        raise InputError(f"El modelo está calibrado solo para {MODEL_LINE}.")
    return std[std["line"] == line].set_index("date").sort_index()[D.HOUR_COLS]


def merge_context(history: pd.DataFrame, new: pd.DataFrame | None) -> pd.DataFrame:
    """Contexto = histórico + días cargados (los cargados reemplazan fechas repetidas)."""
    if new is None or new.empty:
        return history[D.HOUR_COLS]
    ctx = pd.concat([history[D.HOUR_COLS].drop(index=new.index, errors="ignore"), new[D.HOUR_COLS]]).sort_index()
    return ctx.reindex(pd.date_range(ctx.index.min(), ctx.index.max(), freq="D"))


def _hour_z_threshold(det, thr: float) -> float:
    if det.agg == "peak_shift":
        st = det.agg_stats_
        return float(st["peak_med"] + thr * st["peak_mad"])
    return float(thr)


def score_days(det, thr: float, context: pd.DataFrame, dates) -> list[DayResult]:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    missing = [d for d in dates if d not in context.index or context.loc[d, D.SCORED_HOURS].isna().all()]
    if missing:
        raise InputError(f"No hay datos para: {', '.join(str(d.date()) for d in missing[:5])}.")
    lvl = level_series(context, dates, det.exclude_season)
    no_hist = [d for d in dates if np.isnan(lvl.loc[d])]
    if no_hist:
        raise InputError(
            f"Historia insuficiente para {', '.join(str(d.date()) for d in no_hist[:5])}: se requieren al menos "
            f"{LEVEL_MIN_DAYS} días hábiles/sábados completos en los {max(LEVEL_FALLBACK_DAYS)} días previos. "
            "Incluya en el archivo las semanas anteriores.")
    X = context.loc[dates, D.SCORED_HOURS]
    sc = det.score(X, context)
    e = det.expected_log(dates, context)
    sig = det._sigma(dates)
    zt = _hour_z_threshold(det, thr)
    out = []
    for d in dates:
        z = sc.z.loc[d]
        obs = X.loc[d]
        miss = [c for c in D.SCORED_HOURS if np.isnan(obs[c])]
        dev = float(np.expm1(np.nanmean(np.log1p(obs) - e.loc[d])))
        s = float(sc.day.loc[d])
        out.append(DayResult(
            date=d, score=s, threshold=thr, alert=bool(s > thr), day_type=str(day_type_ext([d])[0]), note=event_note(d),
            observed=obs, expected=np.expm1(e.loc[d]), band_low=np.expm1(e.loc[d] - zt * sig.loc[d]),
            band_high=np.expm1(e.loc[d] + zt * sig.loc[d]), z=z,
            hour_flags=[c for c in D.SCORED_HOURS if abs(z[c]) >= zt], deviation_pct=100 * dev, missing_hours=miss))
    return out


def summary_table(results: list[DayResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Fecha": r.date.date(), "Estado": r.status, "Score": round(r.score, 2), "Umbral": round(r.threshold, 2),
        "Desviación media %": round(r.deviation_pct, 1),
        "Horas fuera de banda": ", ".join(f"{h[1:]}h" for h in r.hour_flags),
        "Tipo de día": r.day_type, "Calendario": r.note or "",
        "Horas sin dato": ", ".join(f"{h[1:]}h" for h in r.missing_hours),
    } for r in results])
