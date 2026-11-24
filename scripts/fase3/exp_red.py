"""F10.3 — Experimentos a nivel de red: H1 (coherencia entre líneas), H2 (local vs. sistémica),
H3 (redistribución, descriptivo) y H4 (fallas de datos de red).

Salida: reports/fase3/red.md y red.json
Uso: python scripts/fase3/exp_red.py [--rapido]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import data as D
from metro_anomaly.detectors import ProfileContextDetector
from metro_anomaly.evaluation import HOLDOUT, budget_threshold, metrics, paired_recall_diff, rolling_evaluate
from metro_anomaly.injection import SCENARIO_BY_ID, inject
from metro_anomaly.red import CoherenceDetector, NetworkDataMonitor, classify_scope, fit_line_profiles, network_z

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "fase3"
MAIN = ["LINEA 1", "LINEA 2", "LINEA A", "LINEA B", "LINEA H", "LINEA J", "LINEA K", "LINEA M", "LINEA O", "LINEA P", "LINEA T-A"]
PERIODS = {"dev": dict(train_start="2022-01-01", eval_start="2022-04-01", eval_end="2023-06-30"),
           "holdout": dict(train_start="2022-01-01", eval_start=HOLDOUT[0], eval_end=HOLDOUT[1])}
H1_LINES = ["LINEA A", "LINEA B", "LINEA 1", "LINEA T-A"]  # A: criterio pre-registrado; las demás: secundario


def h1(Ws, periods):
    out = {}
    for ln in H1_LINES:
        W = Ws[ln]
        for per in periods:
            kw = PERIODS[per]
            rb = rolling_evaluate(lambda: ProfileContextDetector(agg="peak_shift"), W, **kw)
            rc = rolling_evaluate(lambda: CoherenceDetector(ln, Ws, train_start=kw["train_start"]), W, **kw)
            tb, tc = budget_threshold(rb.real["score"]), budget_threshold(rc.real["score"])
            mb, mc = metrics(rb, tb), metrics(rc, tc)
            d = paired_recall_diff(rc, tc, rb, tb, "moderada", n_boot=500)
            keep = ["tasa_alertas", "tasa_alertas_festivos", "recall_moderada", "recall_severa", "recall_leve",
                    "recall_moderada_media", "localizacion_hora", "recall_T1_moderada", "recall_T2_moderada", "recall_T3_moderada"]
            out[f"{ln}|{per}"] = {"perfil_contextual": {k: round(mb.get(k, np.nan), 3) for k in keep},
                                  "coherencia_red": {k: round(mc.get(k, np.nan), 3) for k in keep},
                                  "diferencia_recall_moderada": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}}
            # alertas reales de cada uno en el periodo (para revisar qué días cambian)
            out[f"{ln}|{per}"]["dias_alerta_solo_perfil"] = [str(x.date()) for x in rb.real.index[(rb.real["score"] > tb) & ~(rc.real["score"].reindex(rb.real.index) > tc)]][:30]
            out[f"{ln}|{per}"]["dias_alerta_solo_coherencia"] = [str(x.date()) for x in rc.real.index[(rc.real["score"] > tc) & ~(rb.real["score"].reindex(rc.real.index) > tb)]][:30]
            print("H1", ln, per, out[f"{ln}|{per}"]["diferencia_recall_moderada"], flush=True)
    return out


def h2_h3(Ws):
    """Inyecciones locales (solo A) y sistémicas (mismo factor en todas las líneas) en el holdout."""
    target = "LINEA A"
    rows = []
    real_rows = []
    scen = [f"{t}_{s}" for t in ("T1", "T2", "T3") for s in ("severa", "moderada")]
    for ms in pd.date_range(HOLDOUT[0], HOLDOUT[1], freq="MS"):
        me = ms + pd.offsets.MonthEnd(0)
        dets = fit_line_profiles(Ws, "2022-01-01", ms - pd.Timedelta(days=1))
        days = Ws[target].loc[ms:me]
        days = days[days.notna().all(axis=1)]
        z_real = network_z(Ws, dets, days.index)
        others_real = {ln: z for ln, z in z_real.items() if ln != target}
        for t in days.index:  # días reales con alguna hora |z| > 3 en A
            zt = z_real[target].loc[t]
            if (zt.abs() > 3).any():
                r = classify_scope(zt, {ln: z.loc[t] for ln, z in others_real.items()})
                real_rows.append({"fecha": t, **r})
        for k, sid in enumerate(scen):
            seed = int(ms.strftime("%Y%m")) * 31 + k
            Xi, mask, ok = inject(days, sid, seed, context=Ws[target])
            ratio = (Xi / days.replace(0, np.nan)).fillna(1.0)
            zA_loc = dets[target]._z(Xi, Ws[target])
            # sistémica: el mismo factor, en las mismas horas, en todas las líneas
            z_sys = {}
            for ln, det in dets.items():
                if ln == target:
                    continue
                W = Ws[ln][D.SCORED_HOURS]
                Xo = W.reindex(days.index)
                z_sys[ln] = det._z(Xo * ratio.reindex(columns=D.SCORED_HOURS), W)
            for t in days.index:
                if not ok.loc[t]:
                    continue
                zl = zA_loc.loc[t]
                if not (zl.abs() > 3).any():
                    continue  # sin desviación visible en A: nada que clasificar
                rl = classify_scope(zl, {ln: z.loc[t] for ln, z in others_real.items()})
                rs = classify_scope(zl, {ln: z.loc[t] for ln, z in z_sys.items()})
                rows.append({"fecha": t, "escenario": sid, "verdad": "local", "pred": rl["alcance"], "frac": rl["fraccion_lineas"]})
                rows.append({"fecha": t, "escenario": sid, "verdad": "generalizada", "pred": rs["alcance"], "frac": rs["fraccion_lineas"]})
        print("H2", ms.date(), flush=True)
    R = pd.DataFrame(rows)
    acc = float((R["verdad"] == R["pred"]).mean())
    conf = pd.crosstab(R["verdad"], R["pred"]).to_dict()
    by_scn = R.assign(ok=R["verdad"] == R["pred"]).groupby(["escenario", "verdad"])["ok"].mean().round(3).unstack().to_dict()
    RR = pd.DataFrame(real_rows)
    # Días reales específicos
    special = {}
    for d, hs in (("2023-10-29", None), ("2023-08-28", ["h17", "h18", "h19"]), ("2023-08-22", ["h07", "h08", "h09"])):
        t = pd.Timestamp(d)
        dets = fit_line_profiles(Ws, "2022-01-01", t.replace(day=1) - pd.Timedelta(days=1))
        z = network_z(Ws, dets, [t])
        zt = z[target].loc[t] if hs is None else z[target].loc[t].where(z[target].columns.isin(hs))
        r = classify_scope(zt.dropna() if hs else zt, {ln: zz.loc[t] for ln, zz in z.items() if ln != target})
        special[d] = {**r, "z_por_linea": {ln: {h: (None if np.isnan(v) else round(float(v), 1)) for h, v in zz.loc[t].items()
                                                if hs is None or h in hs} for ln, zz in z.items()}}
    # H3: en 2023-08-28 17–19 h, líneas con z > 3 (aumento) — perfil contextual de cada línea
    z828 = special["2023-08-28"]["z_por_linea"]
    h3 = {ln: v for ln, v in z828.items() if any((x is not None and x > 3) for x in v.values())}
    return {"exactitud": round(acc, 3), "n": int(len(R)), "confusion": conf, "exactitud_por_escenario": by_scn,
            "dias_reales_alcance": RR["alcance"].value_counts().to_dict() if len(RR) else {},
            "dias_reales_detalle": [{"fecha": str(r.fecha.date()), "alcance": r.alcance, "fraccion": round(r.fraccion_lineas, 2),
                                     "lineas": r.lineas_acompanan, "sentido": r.sentido} for r in RR.itertuples()] if len(RR) else [],
            "casos": special}, {"lineas_con_aumento_z>3_17_19h": h3}


def h4(Ws, start="2022-07-01", end="2024-06-30", n=400, seed=0):
    mon = NetworkDataMonitor(Ws, start, end)
    dates = pd.date_range(start, end)
    real = mon.flags(dates)
    rate_days = real["fecha"].nunique() / len(dates)
    rng = np.random.default_rng(seed)
    res = {"red_k>=3": [], "una_o_dos_lineas": []}
    for i in range(n):
        t = dates[rng.integers(len(dates))]
        h = D.SCORED_HOURS[rng.integers(2, 17)]  # 06–20 h
        hab = mon.habitual(t, h)
        if len(hab) < 6:
            continue
        net = i % 2 == 0
        k = int(rng.integers(3, 7)) if net else int(rng.integers(1, 3))
        pick = list(rng.choice(hab, size=k, replace=False))
        val = 0.0 if rng.random() < 0.5 else np.nan
        f = mon.failing(t, h, {ln: val for ln in pick})
        res["red_k>=3" if net else "una_o_dos_lineas"].append(len(f) >= mon.min_lines)
    return {"dias_reales_con_alerta": int(real["fecha"].nunique()), "dias_evaluados": len(dates),
            "tasa_dias_reales": round(rate_days, 4), "recall_inyeccion_red": round(float(np.mean(res["red_k>=3"])), 3),
            "n_inyeccion_red": len(res["red_k>=3"]),
            "falsas_alertas_red_con_1_2_lineas": round(float(np.mean(res["una_o_dos_lineas"])), 3),
            "n_inyeccion_1_2": len(res["una_o_dos_lineas"]),
            "alertas_reales": [{"fecha": str(r.fecha.date()), "hora": r.hora, "n": r.n_lineas, "lineas": r.lineas} for r in real.itertuples()]}


def main():
    rapido = "--rapido" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    df, _ = D.load_all(ROOT / "data")
    Ws = {ln: D.line_series(df, ln)[D.SCORED_HOURS] for ln in MAIN}
    res = {}
    res["H4"] = h4(Ws)
    print("H4", {k: v for k, v in res["H4"].items() if k != "alertas_reales"}, flush=True)
    res["H2"], res["H3"] = h2_h3(Ws)
    print("H2", res["H2"]["exactitud"], res["H2"]["confusion"], flush=True)
    res["H1"] = h1(Ws, ["holdout"] if rapido else ["dev", "holdout"])
    (OUT / "red.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = ["# Experimentos de red: H1–H4 (generado por `scripts/fase3/exp_red.py`)", ""]
    md += ["## H1 — Detector de coherencia entre líneas vs. perfil contextual (mismo presupuesto: 5 % de días reales)", ""]
    for k, v in res["H1"].items():
        t = pd.DataFrame({"perfil_contextual": v["perfil_contextual"], "coherencia_red": v["coherencia_red"]})
        md += [f"### {k}", "", t.to_markdown(), "", f"Diferencia pareada de recall moderado (coherencia − perfil): {v['diferencia_recall_moderada']}", ""]
    h2r = res["H2"]
    md += ["## H2 — Alcance local vs. generalizado (inyecciones en el holdout)", "",
           f"Exactitud: **{h2r['exactitud']}** (n = {h2r['n']})", "", pd.DataFrame(h2r["confusion"]).fillna(0).astype(int).to_markdown(), "",
           pd.DataFrame(h2r["exactitud_por_escenario"]).to_markdown(), "",
           f"Días reales del holdout con alguna hora |z| > 3 en la Línea A, por alcance: {h2r['dias_reales_alcance']}", ""]
    for d, c in h2r["casos"].items():
        md += [f"- **{d}**: alcance = {c['alcance']} (fracción {c['fraccion_lineas']:.2f}, {c['sentido']}); acompañan: {', '.join(c['lineas_acompanan']) or '—'}"]
    md += ["", "## H3 — Redistribución el 2023-08-28 (17–19 h)", "", pd.DataFrame(res["H2"]["casos"]["2023-08-28"]["z_por_linea"]).T.to_markdown(), "",
           f"Líneas con aumento z > 3: {list(res['H3']['lineas_con_aumento_z>3_17_19h'])}", ""]
    h4r = res["H4"]
    md += ["## H4 — Fallas de datos a nivel de red", "",
           f"- Días reales (2022-07…2024-06) con ≥ 3 líneas habituales sin dato o en cero a la misma hora: {h4r['dias_reales_con_alerta']} de {h4r['dias_evaluados']} ({100 * h4r['tasa_dias_reales']:.2f} %)",
           f"- Recall con fallas simuladas en 3–6 líneas: {h4r['recall_inyeccion_red']} (n = {h4r['n_inyeccion_red']})",
           f"- Alertas de red cuando fallan solo 1–2 líneas (deben ser 0): {h4r['falsas_alertas_red_con_1_2_lineas']} (n = {h4r['n_inyeccion_1_2']})", "",
           "Alertas reales:", ""] + [f"- {a['fecha']} {a['hora']}: {a['n']} líneas ({a['lineas']})" for a in h4r["alertas_reales"][:60]]
    (OUT / "red.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)[:6000])


if __name__ == "__main__":
    main()
