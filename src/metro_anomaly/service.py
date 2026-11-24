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
from .calendar import ELECTION_DAYS, ELECTION_FARE_FREE_NOTE, day_type_ext, event_note
from .detectors import LEVEL_FALLBACK_DAYS, LEVEL_MIN_DAYS, level_series

MODEL_LINE = "LINEA A"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
#: Reglas de calidad de datos (no estadísticas):
#: - cero sospechoso: 0 entradas en una hora para la que se esperaban ≥ 1.000;
#: - valor implausible: más de 1,5 veces el máximo histórico de esa hora (datos anteriores al día evaluado).
#: Ajuste F8: la regla anterior ("> 5× lo esperado") marcaba como error el 08/12/2023 22 h (7.918 entradas en un
#: festivo de temporada, dentro del rango histórico y similar al 08/12/2022), es decir, un patrón operativo real.
DQ_ZERO_MIN_EXPECTED = 1000
DQ_IMPLAUSIBLE_FACTOR = 1.5
DAY_TYPE_LABELS = {"habil": "Día hábil", "sabado": "Sábado", "domingo_festivo": "Domingo o festivo",
                   "especial": "Víspera de Navidad/Año Nuevo", "habil_temporada": "Día hábil en temporada",
                   "sabado_temporada": "Sábado en temporada"}
CATEGORY_LABELS = {"sin_anomalia": "Sin anomalía estadística", "anomalia_estadistica": "Anomalía estadística",
                   "posible_problema_de_datos": "Posible problema de datos",
                   "efecto_de_medicion": "Efecto de medición conocido"}


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
    dq_flags: list[str] = field(default_factory=list)
    severity: str | None = None
    exceedance: float | None = None  # fracción de días de referencia con score ≥ este score

    @property
    def category(self) -> str:
        if any(f.startswith("Jornada electoral") for f in self.dq_flags):
            return "efecto_de_medicion"
        if any(not f.startswith("Sin dato") for f in self.dq_flags):
            return "posible_problema_de_datos"
        return "anomalia_estadistica" if self.alert else "sin_anomalia"

    @property
    def status(self) -> str:
        return CATEGORY_LABELS[self.category]

    @property
    def day_type_label(self) -> str:
        return DAY_TYPE_LABELS.get(self.day_type, self.day_type)


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


def severity_of(score: float, thr: float, ref: np.ndarray | None) -> tuple[str | None, float | None]:
    """Severidad empírica respecto de los scores de referencia (días reales de DEV, fuera de muestra).
    moderada: umbral < score ≤ p99 · alta: p99 < score ≤ máximo de referencia · extrema: > máximo."""
    if ref is None or len(ref) == 0:
        return (None, None)
    exc = float((ref >= score).mean())
    if score <= thr:
        return (None, exc)
    p99, mx = float(np.quantile(ref, 0.99)), float(ref.max())
    return ("moderada" if score <= p99 else "alta" if score <= mx else "extrema", exc)


def data_quality_flags(obs: pd.Series, expected: pd.Series, hist_max: pd.Series | None = None) -> list[str]:
    flags = [f"Sin dato a las {c[1:]} h" for c in D.SCORED_HOURS if np.isnan(obs[c])]
    zeros = [c for c in D.SCORED_HOURS if obs[c] == 0 and expected[c] >= DQ_ZERO_MIN_EXPECTED]
    if zeros:
        flags.append("Cero sospechoso (posible falla de captura o cierre total) a las " + ", ".join(f"{c[1:]} h" for c in zeros))
    if hist_max is not None:
        hi = [c for c in D.SCORED_HOURS if not np.isnan(hist_max[c]) and obs[c] > DQ_IMPLAUSIBLE_FACTOR * hist_max[c]]
        if hi:
            flags.append("Valor implausible (más de 1,5 veces el máximo histórico de la hora; posible error de registro o de unidad) a las "
                         + ", ".join(f"{c[1:]} h" for c in hi))
    return flags


def score_days(det, thr: float, context: pd.DataFrame, dates, ref: np.ndarray | None = None) -> list[DayResult]:
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
        sev, exc = severity_of(s, thr, ref)
        expd = np.expm1(e.loc[d])
        out.append(DayResult(
            date=d, score=s, threshold=thr, alert=bool(s > thr), day_type=str(day_type_ext([d])[0]), note=event_note(d),
            observed=obs, expected=np.expm1(e.loc[d]), band_low=np.expm1(e.loc[d] - zt * sig.loc[d]),
            band_high=np.expm1(e.loc[d] + zt * sig.loc[d]), z=z,
            hour_flags=[c for c in D.SCORED_HOURS if abs(z[c]) >= zt], deviation_pct=100 * dev, missing_hours=miss,
            dq_flags=data_quality_flags(obs, expd, context.loc[: d - pd.Timedelta(days=1), D.SCORED_HOURS].max())
            + ([ELECTION_FARE_FREE_NOTE] if str(d.date()) in ELECTION_DAYS else []),
            severity=sev, exceedance=exc))
    return out


def score_series(det, thr: float, context: pd.DataFrame, start, end) -> pd.DataFrame:
    """Scores diarios de un rango (para la línea de tiempo). Omite días sin datos o sin historia suficiente."""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    idx = idx[idx.isin(context.index)]
    idx = idx[context.loc[idx, D.SCORED_HOURS].notna().any(axis=1).to_numpy()]
    lvl = level_series(context, idx, det.exclude_season)
    idx = idx[lvl.notna().to_numpy()]
    if len(idx) == 0:
        return pd.DataFrame(columns=["score", "alerta"])
    sc = det.score(context.loc[idx, D.SCORED_HOURS], context).day
    return pd.DataFrame({"score": sc, "alerta": sc > thr})


def summary_table(results: list[DayResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Fecha": r.date.strftime("%d/%m/%Y"), "Estado": r.status, "Severidad": r.severity or "",
        "Score": round(r.score, 2), "Umbral": round(r.threshold, 2),
        "Frecuencia histórica %": None if r.exceedance is None else round(100 * r.exceedance, 1),
        "Desviación media %": round(r.deviation_pct, 1),
        "Horas fuera de banda": ", ".join(f"{h[1:]} h" for h in r.hour_flags),
        "Tipo de día": r.day_type_label, "Calendario": r.note or "",
        "Calidad de datos": "; ".join(r.dq_flags),
    } for r in results])


def reference_scores(doc: dict) -> np.ndarray | None:
    ref = doc.get("reference_scores", {}).get("valores_ordenados")
    return None if ref is None else np.asarray(ref, dtype=float)
