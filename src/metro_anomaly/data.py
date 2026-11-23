"""Carga, validación y normalización de los archivos oficiales de afluencia del Metro de Medellín.

Formato de entrada esperado (un archivo por año, ``Afluencia_Metro_AAAA.xlsx``):
``Día`` (dd/mm/aaaa o fecha), ``Línea de Servicio``, 20 columnas horarias 04:00…23:00
(tipo ``datetime.time`` en Excel o texto ``HH:MM:SS`` en CSV) y ``Total general (Número de pasajeros)``.

Salida canónica: tabla larga por (fecha, línea) con columnas ``h04``…``h23`` (float, NaN = sin dato)
y ``total``.
"""
from __future__ import annotations

import datetime as dt
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = list(range(4, 24))
HOUR_COLS = [f"h{h:02d}" for h in HOURS]
#: Horas usadas para puntuar. 23:00 se excluye: es un tramo residual de cierre (media ≈ 95 pasajeros
#: frente a miles en las demás) y tiene un 34 % de faltantes en la Línea A (ver docs/BITACORA.md, F2.3).
SCORED_HOURS = HOUR_COLS[:-1]
DATE_COL, LINE_COL, TOTAL_COL = "Día", "Línea de Servicio", "Total general (Número de pasajeros)"


class DataValidationError(ValueError):
    """Error de esquema o contenido en un archivo de afluencia."""


def normalize_line(name: str) -> str:
    """'Línea a ' -> 'LINEA A' (sin tildes, mayúsculas, espacios simples)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def _hour_of(col) -> int | None:
    if isinstance(col, dt.time):
        return col.hour if col.minute == 0 and col.second == 0 else None
    if isinstance(col, str):
        try:
            t = dt.time.fromisoformat(col.strip())
        except ValueError:
            return None
        return t.hour if t.minute == 0 and t.second == 0 else None
    return None


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    out = pd.to_datetime(s.astype(str).str.strip(), format="%d/%m/%Y", errors="coerce")
    if out.isna().any():  # admite también ISO (AAAA-MM-DD), p. ej. en CSV exportados
        iso = pd.to_datetime(s[out.isna()].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")
        out = out.fillna(iso)
    bad = s[out.isna()]
    if len(bad):
        raise DataValidationError(f"Fechas no reconocidas (se espera dd/mm/aaaa): {bad.head(3).tolist()}")
    return out


def standardize(raw: pd.DataFrame, source: str = "<memoria>") -> pd.DataFrame:
    """Valida un DataFrame con el formato oficial y lo convierte al formato canónico."""
    missing = [c for c in (DATE_COL, LINE_COL) if c not in raw.columns]
    if missing:
        raise DataValidationError(f"{source}: faltan columnas obligatorias {missing}")
    hour_map = {c: _hour_of(c) for c in raw.columns}
    hour_map = {c: h for c, h in hour_map.items() if h is not None}
    found = sorted(hour_map.values())
    if found != HOURS:
        falt = sorted(set(HOURS) - set(found))
        extra = sorted(set(found) - set(HOURS))
        raise DataValidationError(f"{source}: columnas horarias inválidas (faltan {falt}, sobran {extra})")
    out = pd.DataFrame({"date": _parse_dates(raw[DATE_COL]), "line": raw[LINE_COL].map(normalize_line)})
    for c, h in hour_map.items():
        v = pd.to_numeric(raw[c], errors="coerce")
        nonnum = v.isna() & raw[c].notna()
        if nonnum.any():
            raise DataValidationError(f"{source}: valores no numéricos en {h:02d}:00 (filas {list(raw.index[nonnum][:3])})")
        out[f"h{h:02d}"] = v.astype(float)
    if (out[HOUR_COLS] < 0).any().any():
        raise DataValidationError(f"{source}: hay conteos negativos")
    out["total"] = pd.to_numeric(raw[TOTAL_COL], errors="coerce") if TOTAL_COL in raw.columns else np.nan
    dup = out.duplicated(["date", "line"], keep=False)
    if dup.any():
        raise DataValidationError(f"{source}: filas duplicadas para (fecha, línea): {out.loc[dup, ['date', 'line']].head(3).values.tolist()}")
    out["source"] = source
    return out[["date", "line", *HOUR_COLS, "total", "source"]]


def read_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise DataValidationError(f"Extensión no soportada: {path.suffix}")
    return standardize(raw, source=path.name)


@dataclass
class LoadReport:
    files: list[str] = field(default_factory=list)
    cross_file_duplicates: int = 0
    conflicting_duplicates: int = 0
    dropped: list[tuple] = field(default_factory=list)


def resolve_duplicates(df: pd.DataFrame, report: LoadReport | None = None) -> pd.DataFrame:
    """Duplicados (fecha, línea) entre archivos. Regla determinista: se conserva la fila del archivo
    cuyo año coincide con el año de la fecha; si no hay ninguno, la del primer archivo en orden alfabético.
    Se informa si los valores de las copias difieren."""
    report = report or LoadReport()
    d = df.copy()
    d["_own"] = d.apply(lambda r: str(r["date"].year) in r["source"], axis=1)
    d = d.sort_values(["date", "line", "_own", "source"], ascending=[True, True, False, True])
    dup_mask = d.duplicated(["date", "line"], keep="first")
    if dup_mask.any():
        keys = d.loc[dup_mask, ["date", "line"]]
        for _, k in keys.iterrows():
            g = d[(d.date == k.date) & (d.line == k.line)]
            if not np.allclose(g[HOUR_COLS].to_numpy(float), g[HOUR_COLS].iloc[[0]].to_numpy(float), equal_nan=True):
                report.conflicting_duplicates += 1
            report.dropped.append((str(k.date.date()), k.line, g["source"].iloc[1:].tolist()))
        report.cross_file_duplicates += int(dup_mask.sum())
    return d.loc[~dup_mask].drop(columns="_own").sort_values(["line", "date"]).reset_index(drop=True)


def load_all(data_dir: str | Path, years=range(2019, 2025)) -> tuple[pd.DataFrame, LoadReport]:
    report = LoadReport()
    parts = []
    for y in years:
        p = Path(data_dir) / f"Afluencia_Metro_{y}.xlsx"
        if not p.exists():
            raise DataValidationError(f"No existe {p}")
        parts.append(read_file(p))
        report.files.append(p.name)
    return resolve_duplicates(pd.concat(parts, ignore_index=True), report), report


def line_series(df: pd.DataFrame, line: str = "LINEA A", start=None, end=None) -> pd.DataFrame:
    """Tabla ancha diaria (índice = calendario completo) de una línea. Días sin registro -> NaN y
    ``missing_day=True``. No imputa: la imputación con información futura era una fuente de fuga (auditoría)."""
    line = normalize_line(line)
    d = df[df["line"] == line].set_index("date").sort_index()
    if d.empty:
        raise DataValidationError(f"No hay datos para la línea {line}")
    idx = pd.date_range(start or d.index.min(), end or d.index.max(), freq="D", name="date")
    w = d.reindex(idx)[[*HOUR_COLS, "total"]]
    w["missing_day"] = w[HOUR_COLS].isna().all(axis=1)
    return w


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de calidad por (año, línea)."""
    rows = []
    for (year, line), g in df.groupby([df["date"].dt.year, "line"]):
        days_expected = pd.Timestamp(year, 12, 31).dayofyear if year != df["date"].dt.year.max() else None
        hsum = g[HOUR_COLS].sum(axis=1, min_count=1)
        rows.append({
            "año": year, "línea": line, "días": len(g),
            "días_calendario": days_expected,
            "nan_h22_pct": round(100 * g["h22"].isna().mean(), 1),
            "nan_h23_pct": round(100 * g["h23"].isna().mean(), 1),
            "nan_04_21_pct": round(100 * g[HOUR_COLS[:-2]].isna().mean().mean(), 2),
            "ceros_04_22_pct": round(100 * (g[SCORED_HOURS] == 0).mean().mean(), 2),
            "total_consistente_pct": round(100 * np.isclose(hsum, g["total"]).mean(), 1),
            "media_diaria": round(float(hsum.mean()), 0),
        })
    return pd.DataFrame(rows)
