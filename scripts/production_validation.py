"""F8 — Validación de preparación para producción (simulación cronológica y escenarios controlados).

Periodo simulado: HOLDOUT 2023-07-01…2024-06-30 (ya usado en F4; aquí se valida la OPERACIÓN del sistema,
no se selecciona ningún modelo). Política: reentrenamiento mensual, umbral fijo del artefacto, fin de día e
intra-día (horario). Escribe reports/produccion/validacion_produccion.md y .json.

Uso: python scripts/production_validation.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import data as D
from metro_anomaly.artifacts import load_profile
from metro_anomaly.calendar import day_type, is_holiday
from metro_anomaly.data import SCORED_HOURS
from metro_anomaly.simulation import (HOURS_INT, batch_scores, end_of_day, factory_for, first_detection,
                                      intraday_scores, monthly_models)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "produccion"
START, END = pd.Timestamp("2023-07-01"), pd.Timestamp("2024-06-30")
H = {int(c[1:]): c for c in SCORED_HOURS}
RES: dict = {}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def log(k, v):
    v = _clean(v)
    RES[k] = v
    print(f"[{k}] {json.dumps(v, default=str, ensure_ascii=False)[:400]}", flush=True)


def complete_days(W, a=START, b=END):
    d = W.loc[a:b]
    return d.index[d[SCORED_HOURS].notna().all(axis=1)]


# ----------------------------------------------------------------------------------------- E1/E2
def e1_eod(W, models, thr, fac):
    days = complete_days(W)
    g = end_of_day(W, models, days, guarded=True)
    b = batch_scores(W, models, W.loc[days])["score"]
    ref = pickle.load(open(ROOT / "reports/runs/holdout/perfil_contextual__peak_shift__s0.pkl", "rb")).real["score"] \
        if (ROOT / "reports/runs/holdout/perfil_contextual__peak_shift__s0.pkl").exists() else None
    t0 = time.perf_counter(); _ = monthly_models(W, START, END, fac); fit_s = time.perf_counter() - t0
    alerts = g.scores[g.scores > thr]
    log("E1_fin_de_dia", {
        "dias": len(days), "violaciones_guarda": g.guard_violations,
        "guardado_igual_a_lote": bool(np.allclose(g.scores.values, b.loc[g.scores.index].values, rtol=0, atol=1e-12)),
        "igual_a_evaluacion_F4": None if ref is None else bool(np.allclose(g.scores.values, ref.loc[g.scores.index].values, rtol=0, atol=1e-12)),
        "seg_por_dia_guardado": round(g.seconds_per_day, 4), "seg_reentrenamientos_12_meses": round(fit_s, 2),
        "alertas": len(alerts), "fechas_alerta": [str(d.date()) for d in alerts.index],
        "alertas_por_mes": (g.scores > thr).groupby(g.scores.index.to_period("M")).sum().astype(int).astype(str).to_dict()})
    return g.scores


def e2_stability(W, models, thr, fac, eod):
    nxt = monthly_models(W, START + pd.offsets.MonthBegin(1), END + pd.offsets.MonthBegin(1), fac)
    days = [d for d in eod.index if (d.to_period("M") + 1) in nxt]
    re = pd.Series({d: float(nxt[d.to_period("M") + 1].score(W.loc[[d]], W).day.iloc[0]) for d in days})
    a, b = eod.loc[days] > thr, re > thr
    log("E2_estabilidad", {"dias": len(days), "concordancia_decision": round(float((a == b).mean()), 4),
                           "alertas_originales": int(a.sum()), "alertas_reevaluadas": int(b.sum()),
                           "cambian_de_decision": [str(d.date()) for d in a.index[a != b]],
                           "corr_scores": round(float(np.corrcoef(eod.loc[days], re)[0, 1]), 4)})


# ----------------------------------------------------------------------------------------- E3 intra-día
def inject_intraday(X, kind, rng):
    Y = X.copy()
    onset = pd.Series(rng.integers(6, 21, size=len(X)), index=X.index)
    for t in X.index:
        s = int(onset[t])
        hrs = [h for h in HOURS_INT if h >= s]
        if kind == "abrupta_severa":
            Y.loc[t, [H[h] for h in hrs]] *= 0.2
        elif kind == "abrupta_moderada":
            Y.loc[t, [H[h] for h in hrs]] *= 0.5
        elif kind == "aislada_severa":
            Y.loc[t, H[s]] *= 0.2
        elif kind == "aislada_moderada":
            Y.loc[t, H[s]] *= 0.5
        elif kind == "progresiva_intradia":  # factor de 1,0 en s a 0,3 en 22 h
            for h in hrs:
                Y.loc[t, H[h]] *= 1 - 0.7 * (h - s) / max(22 - s, 1)
        elif kind == "aumento_evento_x1.5":
            Y.loc[t, [H[h] for h in hrs[:3]]] *= 1.5
        elif kind == "aumento_evento_x2":
            Y.loc[t, [H[h] for h in hrs[:3]]] *= 2.0
    return Y, onset


def e3_intraday(W, models, thr, quick):
    days = complete_days(W)
    P = intraday_scores(W, models, W.loc[days])
    cum = (P > thr).cummax(axis=1)
    eod_alert = P[22] > thr
    log("E3a_falsas_alarmas_dia_parcial", {
        "dias": len(days),
        "pct_dias_con_alerta_al_cierre_de_hora": {int(h): round(float(cum[h].mean()) * 100, 1) for h in P.columns},
        "pct_dias_alerta_fin_de_dia": round(float(eod_alert.mean()) * 100, 1),
        "pct_dias_alerta_intradia_que_no_persiste_al_cierre": round(float((cum[22] & ~eod_alert).mean()) * 100, 1)})
    rng = np.random.default_rng(0)
    out = {}
    for kind in ["abrupta_severa", "abrupta_moderada", "aislada_severa", "aislada_moderada", "progresiva_intradia",
                 "aumento_evento_x1.5", "aumento_evento_x2"]:
        Y, onset = inject_intraday(W.loc[days], kind, rng)
        Pi = intraday_scores(W, models, Y)
        det = first_detection(Pi, thr, onset)
        delay = (det - onset + 1).dropna()  # horas desde el inicio de la franja afectada hasta tener el dato que dispara
        r = {"deteccion_en_el_dia": round(float(det.notna().mean()), 3),
             "deteccion_fin_de_dia": round(float((Pi[22] > thr).mean()), 3),
             "retardo_mediana_h": float(np.median(delay)) if len(delay) else None,
             "retardo_p90_h": float(np.percentile(delay, 90)) if len(delay) else None,
             "pct_detectadas_en_<=1h": round(float((delay <= 1).mean()), 3) if len(delay) else None,
             "alarmas_antes_del_inicio(falsas)": round(float(((Pi > thr).apply(lambda r: any(r[h] for h in Pi.columns if h < onset[r.name]), axis=1)).mean()), 3)}
        if kind == "progresiva_intradia":  # ¿detecta antes de que la caída alcance −50 %?
            half = onset + np.ceil((22 - onset) * (0.5 / 0.7))
            r["pct_detectadas_antes_de_alcanzar_-50%"] = round(float(((det < half) & det.notna()).mean()), 3)
        out[kind] = r
    log("E3b_latencia_intradia", out)


# ----------------------------------------------------------------------------------------- E4 multi-día
def simulate_window(Wm, fac, a, b, thr):
    models = monthly_models(Wm, a, b, fac)
    days = complete_days(Wm, a, b)
    return batch_scores(Wm, models, Wm.loc[days])["score"] > thr


def e4_multiday(W, fac, thr):
    starts = [pd.Timestamp(x) for x in ("2023-08-07", "2023-09-04", "2023-10-02", "2024-02-05", "2024-03-04", "2024-04-01")]
    base = simulate_window(W, fac, START, END, thr)
    res = {}

    def run(name, mutate, horizon):
        rows = []
        for s0 in starts:
            Wm = W.copy()
            affected = mutate(Wm, s0)
            win_end = min(s0 + pd.Timedelta(days=horizon), END)
            al = simulate_window(Wm, fac, s0, win_end, thr)
            aff = [d for d in affected if d in al.index]
            first = next((d for d in aff if al.loc[d]), None)
            rows.append({"inicio": str(s0.date()), "dias_afectados": len(aff),
                         "primer_dia_detectado": None if first is None else int((first - s0).days),
                         "pct_dias_afectados_con_alerta": round(float(al.loc[aff].mean()), 3) if aff else None,
                         "alertas_fuera_del_evento": int(al.drop(index=aff, errors="ignore").sum() - base.loc[al.index.difference(aff)].sum())})
        res[name] = {"por_inicio": rows,
                     "deteccion": sum(r["primer_dia_detectado"] is not None for r in rows) / len(rows),
                     "mediana_primer_dia": float(np.median([r["primer_dia_detectado"] for r in rows if r["primer_dia_detectado"] is not None])) if any(r["primer_dia_detectado"] is not None for r in rows) else None,
                     "media_pct_dias_alertados": round(float(np.mean([r["pct_dias_afectados_con_alerta"] for r in rows])), 3)}
        log(f"E4_{name}", {k: v for k, v in res[name].items() if k != "por_inicio"})

    def progressive(Wm, s0):  # −3 %/día durante 14 días y luego estable en −35 % por 30 días
        days = pd.date_range(s0, s0 + pd.Timedelta(days=43))
        for i, d in enumerate(days):
            if d in Wm.index:
                Wm.loc[d, SCORED_HOURS] *= 1 - 0.025 * min(i + 1, 14)
        return list(days)

    def step(f, n=60):
        def m(Wm, s0):
            days = pd.date_range(s0, s0 + pd.Timedelta(days=n - 1))
            Wm.loc[days.intersection(Wm.index), SCORED_HOURS] *= f
            return list(days)
        return m

    def recurrent(Wm, s0):  # cada lunes 17–18 h × 0,5 durante 10 semanas
        days = [s0 + pd.Timedelta(weeks=k) for k in range(10)]
        days = [d for d in days if d in Wm.index and day_type([d])[0] == "habil"]
        Wm.loc[days, ["h17", "h18"]] *= 0.5
        return days

    def seasonal(Wm, s0):  # temporada no modelada: 21 días × 0,85
        days = pd.date_range(s0, s0 + pd.Timedelta(days=20))
        Wm.loc[days.intersection(Wm.index), SCORED_HOURS] *= 0.85
        return list(days)

    run("progresiva_multidia_-2.5%_dia", progressive, 50)
    run("escalon_sostenido_-20%", step(0.8), 75)
    run("escalon_sostenido_+20%", step(1.2), 75)
    run("recurrente_lunes_17-18h_x0.5", recurrent, 75)
    run("estacional_no_modelada_-15%_21d", seasonal, 30)
    # detalle de la recurrente: tasa de detección por ocurrencia
    res_rec = res["recurrente_lunes_17-18h_x0.5"]["por_inicio"]
    return res, base


def e4_contextual(W, models, thr):
    days = complete_days(W)
    dt_ = pd.Series(day_type(days), index=days)
    out = {}
    def med_profile(t, kind):
        win = W.loc[t - pd.Timedelta(days=56): t - pd.Timedelta(days=1)]
        sel = win[pd.Series(day_type(win.index), index=win.index) == kind][SCORED_HOURS].dropna()
        return sel.median()
    for name, target, src in [("perfil_festivo_en_dia_habil", "habil", "domingo_festivo"),
                              ("perfil_habil_en_festivo_o_domingo", "domingo_festivo", "habil"),
                              ("perfil_sabado_en_dia_habil", "habil", "sabado")]:
        sel = days[dt_ == target]
        X = W.loc[sel].copy()
        for t in sel:
            X.loc[t, SCORED_HOURS] = med_profile(t, src).to_numpy()
        s = batch_scores(W, models, X)["score"]
        out[name] = {"dias": len(sel), "deteccion": round(float((s > thr).mean()), 3)}
    log("E4_contextuales", out)


# ----------------------------------------------------------------------------------------- E5 precisión
SINGLE = {
    "T1_moderada": lambda x, r: _block(x, r, 0.5), "T1_severa": lambda x, r: _block(x, r, 0.2),
    "T2_moderada": lambda x, r: x * 0.8, "T3_moderada": lambda x, r: _block(x, r, 1.5, lo=16),
    "T4_cero": lambda x, r: _one(x, r, 0.0), "T4_duplicada": lambda x, r: _one(x, r, 2.0)}


def _block(x, r, f, lo=5):
    k = int(r.integers(1, 4)); s = int(r.integers(lo, 22 - k + 1)); y = x.copy()
    y[[H[h] for h in range(s, s + k)]] *= f; return y


def _one(x, r, f):
    y = x.copy(); h = int(r.integers(6, 21)); y[H[h]] *= f; return y


def e5_precision(W, fac, thr, known_alerts, prevalences=(0.02, 0.05), seeds=range(5)):
    days = complete_days(W)
    out = {}
    for prev in prevalences:
        rows = []
        for seed in seeds:
            r = np.random.default_rng(seed)
            n = max(1, int(round(prev * len(days))))
            inj = pd.DatetimeIndex(sorted(r.choice(days, n, replace=False)))
            Wm = W.copy()
            kinds = r.choice(list(SINGLE), n)
            for t, k in zip(inj, kinds):
                Wm.loc[t, SCORED_HOURS] = SINGLE[k](Wm.loc[t, SCORED_HOURS], r).to_numpy()
            al = simulate_window(Wm, fac, START, END, thr)
            tp = int(al.loc[inj].sum()); fn = len(inj) - tp
            neg = al.drop(index=inj)
            fp = int(neg.sum()); fp_nk = int(neg.drop(index=[d for d in known_alerts if d in neg.index]).sum())
            rows.append({"tp": tp, "fp": fp, "fn": fn, "fp_sin_alertas_reales_conocidas": fp_nk})
        df = pd.DataFrame(rows).sum()
        out[f"prevalencia_{prev}"] = {
            "inyectados": int(df.tp + df.fn), "tp": int(df.tp), "fp": int(df.fp), "fn": int(df.fn),
            "sensibilidad": round(df.tp / (df.tp + df.fn), 3), "precision": round(df.tp / max(df.tp + df.fp, 1), 3),
            "precision_si_las_8_alertas_reales_fueran_eventos": round(df.tp / max(df.tp + df.fp_sin_alertas_reales_conocidas, 1), 3),
            "fp_por_mes": round(df.fp / (12 * len(list(seeds))), 2)}
    log("E5_precision_flujo_mixto", out)


# ----------------------------------------------------------------------------------------- E6 robustez
def e6_robustness(W, models, thr, fac, eod):
    days = complete_days(W)
    out = {}
    rng = np.random.default_rng(1)
    for k in (1, 3, 6):
        X = W.loc[days].copy()
        for t in days:
            X.loc[t, list(rng.choice(SCORED_HOURS, k, replace=False))] = np.nan
        s = batch_scores(W, models, X)["score"]
        out[f"faltan_{k}_horas_aleatorias"] = {"tasa_alertas": round(float((s > thr).mean()), 4),
                                               "tasa_alertas_sin_faltantes": round(float((eod > thr).mean()), 4),
                                               "scores_finitos": bool(np.isfinite(s).all())}
    # dato retrasado: los k días previos aún no llegaron al puntuar t
    for k in (1, 3, 7):
        diffs, flips = [], 0
        for t in days[40:]:
            ctx = W.copy(); ctx.loc[t - pd.Timedelta(days=k): t - pd.Timedelta(days=1), SCORED_HOURS] = np.nan
            s = float(models[t.to_period("M")].score(W.loc[[t]], ctx).day.iloc[0])
            diffs.append(abs(s - eod.loc[t])); flips += int((s > thr) != (eod.loc[t] > thr))
        out[f"retraso_{k}_dias_previos"] = {"dif_abs_media_score": round(float(np.mean(diffs)), 4),
                                            "cambios_de_decision": flips, "dias": len(diffs)}
    # corrupción en un día y contaminación de los 28 días siguientes
    for name, f in [("unidad_x10", 10.0), ("dia_en_ceros", 0.0), ("extremo_x1e4", 1e4)]:
        rows = []
        for s0 in [pd.Timestamp(x) for x in ("2023-08-09", "2023-10-11", "2024-02-14", "2024-04-10")]:
            Wm = W.copy(); Wm.loc[s0, SCORED_HOURS] = Wm.loc[s0, SCORED_HOURS] * f
            al = simulate_window(Wm, fac, s0, min(s0 + pd.Timedelta(days=28), END), thr)
            after = al.loc[s0 + pd.Timedelta(days=1):]
            base_after = (eod.loc[after.index] > thr)
            rows.append({"alerta_dia_corrupto": bool(al.loc[s0]), "alertas_extra_28d": int(after.sum() - base_after.sum())})
        out[f"corrupcion_{name}"] = rows
    # envejecimiento: modelo congelado a 2023-06-30 frente a reentrenamiento mensual
    frozen = fac().fit(W.loc["2022-01-01":"2023-06-30"], W.loc[:"2023-06-30"])
    fz = frozen.score(W.loc[days], W).day
    rng = np.random.default_rng(2)
    Xi = W.loc[days].copy()
    for t in days:
        Xi.loc[t] = _block(Xi.loc[t], rng, 0.5)
    rec_m = batch_scores(W, models, Xi)["score"] > thr
    rec_f = frozen.score(Xi, W).day > thr
    by_m = pd.DataFrame({"alertas_congelado": fz > thr, "alertas_mensual": eod.loc[days] > thr,
                         "recall_T1mod_congelado": rec_f, "recall_T1mod_mensual": rec_m}).groupby(days.to_period("M")).mean().round(3)
    out["envejecimiento_modelo"] = {"por_mes": by_m.astype(str).to_dict(orient="index"),
                                    "tasa_alertas_congelado": round(float((fz > thr).mean()), 4),
                                    "tasa_alertas_mensual": round(float((eod.loc[days] > thr).mean()), 4),
                                    "recall_T1mod_congelado": round(float(rec_f.mean()), 3),
                                    "recall_T1mod_mensual": round(float(rec_m.mean()), 3)}
    log("E6_robustez", out)


def e6_recalibration(W, fac, thr, eod):
    """EXPLORATORIO (el holdout ya se conoce): umbral = cuantil 95 % de los scores de fin de día fuera de muestra
    de los 365 días previos (política definida aquí, no pre-registrada)."""
    dev = pickle.load(open(ROOT / "reports/runs/dev/perfil_contextual__peak_shift__s0.pkl", "rb")).real["score"]
    hist = pd.concat([dev, eod]).sort_index()
    rows = {}
    for ms in pd.date_range(START, END, freq="MS"):
        past = hist.loc[ms - pd.Timedelta(days=365): ms - pd.Timedelta(days=1)]
        t = float(np.quantile(past.dropna(), 0.95))
        cur = eod.loc[ms: ms + pd.offsets.MonthEnd(0)]
        rows[str(ms.to_period("M"))] = {"umbral": round(t, 3), "tasa_alertas": round(float((cur > t).mean()), 3),
                                        "tasa_alertas_umbral_fijo": round(float((cur > thr).mean()), 3)}
    log("E6b_recalibracion_exploratoria", rows)


# ----------------------------------------------------------------------------------------- E7 escalabilidad
def e7_scalability(df, fac):
    out = {}
    t0 = time.perf_counter()
    for line in sorted(df["line"].unique()):
        try:
            W = D.line_series(df, line)
            m = fac().fit(W.loc["2023-01-01":"2023-12-31"], W)
            d = complete_days(W, "2024-01-01", "2024-06-30")
            s = m.score(W.loc[d], W).day
            out[line] = {"dias_puntuados": len(d), "scores_finitos": bool(np.isfinite(s).all())}
        except Exception as e:  # noqa: BLE001
            out[line] = {"error": f"{type(e).__name__}: {e}"}
    out["_segundos_total_12_lineas"] = round(time.perf_counter() - t0, 2)
    log("E7_escalabilidad_multilinea(viabilidad tecnica, NO validez)", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, _ = D.load_all(ROOT / "data")
    W = D.line_series(df, "LINEA A")[SCORED_HOURS]
    _, thr, doc = load_profile(ROOT / "artifacts" / "modelo.json")
    fac = factory_for(doc["detector"], doc["agg"])
    RES["meta"] = {"umbral": thr, "detector": doc["detector"], "agg": doc["agg"], "periodo": [str(START.date()), str(END.date())]}
    models = monthly_models(W, START, END, fac)
    eod = e1_eod(W, models, thr, fac)
    e2_stability(W, models, thr, fac, eod)
    e3_intraday(W, models, thr, a.quick)
    e4_multiday(W, fac, thr)
    e4_contextual(W, models, thr)
    known = [pd.Timestamp(x) for x in RES["E1_fin_de_dia"]["fechas_alerta"]]
    e5_precision(W, fac, thr, known)
    e6_robustness(W, models, thr, fac, eod)
    e6_recalibration(W, fac, thr, eod)
    e7_scalability(df, fac)
    (OUT / "validacion_produccion.json").write_text(json.dumps(RES, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    print("OK", OUT / "validacion_produccion.json")


if __name__ == "__main__":
    main()
