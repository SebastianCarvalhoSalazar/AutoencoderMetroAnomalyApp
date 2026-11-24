"""Plataforma multilínea (F10.4): orquesta fuentes, modelos especializados y reglas sobre la red.

Componentes:
- Fuentes: entradas registradas por línea y hora (dato real), topología verificada (`data/topologia_red.csv`),
  metadatos de líneas (`data/lineas.csv`), calendario y lluvia (solo para investigación; H5b se descartó).
- Modelos: perfil contextual por línea (z por hora), detector de coherencia entre líneas (H1, adoptado en las
  líneas donde mejoró con IC > 0), pronóstico del día siguiente (H5, híbrido elegido en DEV).
- Motor de reglas: prioriza efecto de medición (calendario) > ausencia simultánea en la red > problema de datos > anomalía
  (con alcance descriptivo local / multilínea / generalizada) > sin anomalía.
- Trazabilidad: cada salida lleva el tipo de dato (real, estimación, simulación, predicción) y la versión del modelo.

Nada aquí localiza estaciones, identifica fallas técnicas ni anticipa incidentes: la granularidad es línea × hora.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .calendar import ELECTION_DAYS, event_note
from .detectors import aggregate
from .red import CoherenceDetector, NetworkDataMonitor, classify_scope, fit_line_profiles

MAIN_LINES = ["LINEA A", "LINEA B", "LINEA T-A", "LINEA H", "LINEA J", "LINEA K", "LINEA M", "LINEA P",
              "LINEA 1", "LINEA 2", "LINEA O"]
#: Líneas donde el detector de coherencia mejoró el recall moderado con IC 95 % > 0 (reports/fase3/red.md).
COHERENCE_LINES = ("LINEA A", "LINEA B", "LINEA 1")
#: Modos sin cobro en jornadas electorales según el comunicado oficial del Metro (metro, tranvía y cables).
FARE_FREE_MODES = ("metro", "tranvía", "cable")
DEV_EVAL = ("2022-04-01", "2023-06-30")
#: Líneas donde H6 (GBM base) tuvo AUC > 0,80 y superó al naive con ≥ 10 positivos (reports/fase3/pronostico_holdout.md).
H6_LINES = ["LINEA A", "LINEA 2", "LINEA J", "LINEA M", "LINEA O", "LINEA P", "LINEA T-A"]
SCHEMA_VERSION = 1

TIPOS_DATO = {
    "real": "Dato real: entradas registradas (validaciones pagas) por línea y hora",
    "estimacion": "Estimación: valor esperado y desviación (z) calculados por un modelo con datos anteriores",
    "simulacion": "Simulación: escenario sintético aplicado a datos reales; no ocurrió",
    "prediccion": "Predicción: pronóstico emitido sin conocer el dato del día",
}

CATEGORIAS = {
    "problema_de_datos": "Posible problema de datos",
    "ausencia_simultanea_red": "Ausencia simultánea de registro en varias líneas (causa por determinar)",
    "efecto_de_medicion": "Efecto de medición (jornada sin cobro)",
    "anomalia_local": "Anomalía estadística local",
    "anomalia_multilinea": "Anomalía estadística en varias líneas",
    "anomalia_generalizada": "Anomalía estadística generalizada en la red",
    "sin_anomalia": "Sin anomalía estadística",
    "no_evaluable": "No evaluable (día incompleto en horas que la línea no siempre reporta)",
    "sin_datos": "Sin datos",
}

CATALOGO_MODELOS = [
    {"id": "perfil_contextual", "capacidad": "Desviación por línea y hora (z)", "estado": "adoptado",
     "evidencia": "docs/validacion/INFORME_FINAL.md", "nota": "Modelo base de fases 1–2, ahora aplicado a 11 líneas"},
    {"id": "coherencia_red (H1)", "capacidad": "Anomalía propia de la línea descontando lo que mueve a toda la red",
     "estado": "adoptado (A, B, 1)", "evidencia": "reports/fase3/red.md",
     "nota": "Línea A: +0,198 de recall moderado [0,177; 0,216] con igual presupuesto; T-A sin mejora"},
    {"id": "alcance (H2)", "capacidad": "Local / multilínea / generalizada", "estado": "descriptivo",
     "evidencia": "reports/fase3/red.md", "nota": "Exactitud 0,884 < 0,90 pre-registrado: orientativo, no validado"},
    {"id": "redistribución (H3)", "capacidad": "Líneas que suben cuando otra cae", "estado": "descriptivo",
     "evidencia": "reports/fase3/red.md", "nota": "n = 1 (28/08/2023); sensible al método de referencia; no causal"},
    {"id": "ausencia simultánea (H4)", "capacidad": "≥ 3 líneas habituales sin registro a la vez", "estado": "adoptado",
     "evidencia": "reports/fase3/red.md", "nota": "Recall 1,0 simulado; 4/731 días reales, todos explicados por operación o calendario"},
    {"id": "pronóstico (H5)", "capacidad": "Entradas del día siguiente por línea y hora con intervalo 80 %",
     "estado": "adoptado", "evidencia": "reports/fase3/pronostico_holdout.md",
     "nota": "Híbrido elegido en DEV: WAPE 11,5–27,6 % menor que el naive en las 11 líneas (A: 24,9 %). Intervalo: "
             "sin calibrar falla en A (0,723); con calibración conformal posterior, 0,79–0,82"},
    {"id": "lluvia (H5b)", "capacidad": "Mejorar el pronóstico con lluvia", "estado": "descartado",
     "evidencia": "reports/fase3/pronostico_holdout.md", "nota": "Ni con la lluvia observada (oráculo) mejora el WAPE"},
    {"id": "días de alta demanda (H6)", "capacidad": "Anticipar días > p95 con un día de antelación", "estado": "adoptado (7 líneas)",
     "evidencia": "reports/fase3/pronostico_holdout.md",
     "nota": "AUC > 0,80 y mejor que el naive en A, 2, J, M, O, P, T-A; no en B, K, H; la Línea 1 tiene solo 3 positivos"},
    {"id": "demografía y suelos (H7)", "capacidad": "Contexto territorial", "estado": "descartado",
     "evidencia": "docs/fase3/FUENTES_EXTERNAS.md", "nota": "Anual o estático; no alineable con línea × hora"},
]


@dataclass
class Red:
    Ws: dict  # línea -> DataFrame (fechas × SCORED_HOURS), dato real
    lineas: pd.DataFrame
    topologia: pd.DataFrame


def load_network(data_dir: str | Path) -> Red:
    data_dir = Path(data_dir)
    df, _ = D.load_all(data_dir)
    Ws = {ln: D.line_series(df, ln)[D.SCORED_HOURS] for ln in MAIN_LINES}
    return Red(Ws, pd.read_csv(data_dir / "lineas.csv"), pd.read_csv(data_dir / "topologia_red.csv"))


# ------------------------------------------------------------------------------------------ historia
def build_history(red: Red, start, end, train_start="2022-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducción cronológica causal: para cada mes M se ajustan los modelos con datos < M.

    Devuelve ``(horas, dias)``:
    - horas: fecha, linea, hora, observado (real), esperado, sigma (escala log) y z (estimación), z_coh (líneas H1).
    - dias: fecha, linea, completo, score_perfil, score_coherencia.
    """
    Ws = red.Ws
    hrows, drows = [], []
    for ms in pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), end, freq="MS"):
        me = min(ms + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        ms_eff = max(ms, pd.Timestamp(start))
        # Los modelos se ajustan SIEMPRE con datos anteriores al inicio del mes (no al primer día pedido), para que la
        # operación diaria (que evalúa un solo día a mitad de mes) reproduzca exactamente la reproducción histórica y
        # los umbrales calibrados con ella (defecto D9 corregido; docs/BITACORA.md).
        hist_end = ms - pd.Timedelta(days=1)
        dets = fit_line_profiles(Ws, train_start, hist_end)
        coh = {}
        for ln in COHERENCE_LINES:
            W = Ws[ln]
            coh[ln] = CoherenceDetector(ln, Ws, train_start=train_start, agg="peak_shift").fit(W.loc[train_start:hist_end], W)
        for ln, det in dets.items():
            W = Ws[ln]
            X = W.loc[ms_eff:me]
            X = X[X.notna().any(axis=1)]
            if X.empty:
                continue
            z = det._z(X, W)
            esp = np.expm1(det.expected_log(X.index, W))
            sig = det._sigma(X.index)
            complete = X.notna().all(axis=1)
            sp = pd.Series(np.nan, index=X.index)
            if complete.any() and det.agg_stats_ is not None:
                sp[complete] = aggregate(z[complete], "peak_shift", det.agg_stats_)
            sc = pd.Series(np.nan, index=X.index)
            zc = pd.DataFrame(np.nan, index=X.index, columns=D.SCORED_HOURS)
            if ln in coh and complete.any():
                s = coh[ln].score(X[complete], W)
                sc[complete] = s.day
                zc.loc[complete] = s.z
            for t in X.index:
                drows.append({"fecha": t, "linea": ln, "completo": bool(complete[t]), "score_perfil": sp[t], "score_coherencia": sc[t]})
            long = pd.DataFrame({"fecha": np.repeat(X.index, len(D.SCORED_HOURS)), "linea": ln,
                                 "hora": np.tile([int(c[1:]) for c in D.SCORED_HOURS], len(X)),
                                 "observado": X.to_numpy().ravel(), "esperado": esp.to_numpy().ravel(),
                                 "sigma": sig.to_numpy().ravel(), "z": z.to_numpy().ravel(), "z_coh": zc.to_numpy().ravel()})
            hrows.append(long)
    return pd.concat(hrows, ignore_index=True), pd.DataFrame(drows)


def thresholds_from_dev(dias: pd.DataFrame, budget=0.05) -> dict:
    """Umbral por línea = p95 de su score de detección en días reales de DEV (2022-04…2023-06)."""
    dev = dias[(dias["fecha"] >= DEV_EVAL[0]) & (dias["fecha"] <= DEV_EVAL[1])]
    out = {}
    for ln, g in dev.groupby("linea"):
        col = "score_coherencia" if ln in COHERENCE_LINES else "score_perfil"
        s = g[col].dropna()
        if len(s) >= 60:
            out[ln] = {"score": col, "umbral": float(np.quantile(s, 1 - budget)), "n_dev": int(len(s))}
    return out


# ------------------------------------------------------------------------------------------ reglas
def evaluate_days(red: Red, horas: pd.DataFrame, dias: pd.DataFrame, thr: dict, z_thr: float = 3.0) -> pd.DataFrame:
    """Motor de reglas por línea y día (prioridad: medición > red > datos > anomalía > normal)."""
    mon = NetworkDataMonitor(red.Ws, dias["fecha"].min(), dias["fecha"].max())
    modes = red.lineas.set_index("linea")["modo"].to_dict()
    zpiv = {ln: g.pivot(index="fecha", columns="hora", values="z") for ln, g in horas.groupby("linea")}
    obs = {ln: g.pivot(index="fecha", columns="hora", values="observado") for ln, g in horas.groupby("linea")}
    esp = {ln: g.pivot(index="fecha", columns="hora", values="esperado") for ln, g in horas.groupby("linea")}
    net_flags = mon.flags(sorted(dias["fecha"].unique()))
    net_by_day = net_flags.groupby("fecha")["lineas"].apply(lambda s: set(",".join(s).split(","))).to_dict()
    out = []
    for r in dias.itertuples(index=False):
        t, ln = r.fecha, r.linea
        info = {"fecha": t, "linea": ln, "umbral": np.nan, "score": np.nan, "alcance": None, "lineas_acompanan": "",
                "lineas_sentido_opuesto": "", "horas_fuera": "", "nota_calendario": event_note(t) or ""}
        cfg = thr.get(ln)
        if cfg:
            info["umbral"], info["score"] = cfg["umbral"], getattr(r, cfg["score"])
        o, e = obs[ln].loc[t], esp[ln].loc[t]
        cv, md = mon.cov[ln].loc[t], mon.med[ln].loc[t]
        habit = {h for h in o.index if cv[f"h{h:02d}"] >= 0.9 and md[f"h{h:02d}"] >= mon.min_expected}
        zero = [h for h in o.index if o[h] == 0 and e[h] >= 1000]
        missing = [h for h in o.index if np.isnan(o[h]) and h in habit]  # solo horas que la línea suele reportar
        z = zpiv[ln].loc[t]
        info["horas_fuera"] = ",".join(str(h) for h in z.index[z.abs() > z_thr])
        in_net = ln in net_by_day.get(t, set())
        election = str(t.date()) in ELECTION_DAYS and modes.get(ln) in FARE_FREE_MODES
        if election:
            cat = "efecto_de_medicion"
        elif in_net:
            cat = "ausencia_simultanea_red"
        elif zero or missing:
            cat = "problema_de_datos"
        elif cfg is None or np.isnan(info["score"]):
            cat = "sin_datos" if o.isna().all() else ("no_evaluable" if o.isna().any() else "sin_anomalia")
        elif info["score"] > cfg["umbral"]:
            # en las otras líneas, un cero es ausencia de registro y no cuenta como desviación de demanda
            others = {l2: zpiv[l2].loc[t].where(obs[l2].loc[t] > 0) if t in zpiv[l2].index else pd.Series(np.nan, index=z.index)
                      for l2 in zpiv if l2 != ln}
            sc = classify_scope(z, others, z_thr)
            info["alcance"], info["lineas_acompanan"] = sc["alcance"], ",".join(sc["lineas_acompanan"])
            hrs = [int(h) for h in sc["horas"]]
            sign = -1 if sc["sentido"] == "caida" else 1
            opp = [l2 for l2, zz in others.items()
                   if zz.reindex(hrs).notna().any() and zz.reindex(hrs).mean() * -sign > z_thr]
            info["lineas_sentido_opuesto"] = ",".join(opp)
            cat = {"local": "anomalia_local", "multilinea": "anomalia_multilinea", "generalizada": "anomalia_generalizada"}.get(
                sc["alcance"], "anomalia_local")
        else:
            cat = "sin_anomalia"
        info["categoria"] = cat
        out.append(info)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ trazabilidad
def sha256_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(map(Path, paths)):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_commit(root: Path) -> str | None:
    """Commit del código que construyó los artefactos; '+cambios_sin_commit' si el árbol de código no está limpio."""
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src", "scripts", "pages", "app.py"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+cambios_sin_commit" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(out_dir: Path, root: Path, data_files, extra: dict) -> dict:
    doc = {"schema_version": SCHEMA_VERSION, "creado": pd.Timestamp.now().isoformat(timespec="seconds"),
           "git_commit_build": git_commit(root), "sha256_datos": sha256_files(data_files),
           "tipos_de_dato": TIPOS_DATO, "categorias": CATEGORIAS, "catalogo_modelos": CATALOGO_MODELOS, **extra}
    (out_dir / "manifiesto.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return doc
