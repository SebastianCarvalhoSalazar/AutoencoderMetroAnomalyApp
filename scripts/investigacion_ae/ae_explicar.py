"""Datos para explicar cómo funciona el autoencoder con calendario (`ae_contextual/top2`) y cómo decide.

Usa exactamente los modelos de la evaluación rolling (mismos datos de ajuste y semilla) y el umbral fijado en DEV:
- ejemplos: días reales y días con anomalías simuladas; por hora, la entrada estandarizada, la reconstrucción, el
  residuo z y lo esperado en pasajeros por el autoencoder y por el perfil vigente; el score y la decisión de ambos.
- latente: las 4 coordenadas del cuello de botella de cada día real (proyectadas a 2D con PCA).
- distribucion: scores de días reales (DEV y evaluación) y de días con anomalías simuladas, relativos al umbral.
- mapa: |z| por día y hora del autoencoder en el periodo de evaluación.

Salida: reports/investigacion_ae/explicar.json  (requiere TensorFlow)
Uso: python scripts/investigacion_ae/ae_explicar.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import data as D
from metro_anomaly.calendar import day_type, event_note
from metro_anomaly.detectors import ProfileContextDetector, aggregate
from metro_anomaly.detectors_nn import AEContextDetector
from metro_anomaly.injection import inject

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "investigacion_ae"
RUNS = OUT / "runs"
TRAIN_START = "2022-01-01"
DAY = pd.Timestamp("2024-06-05")  # miércoles normal del último mes evaluado (sin festivo ni evento)
EJEMPLOS = [  # (id, título, fecha, escenario de inyección o None, semilla)
    ("normal", "Día normal (miércoles 05/06/2024)", DAY, None, 0),
    ("caida_horas", "Caída simulada de 2 horas a la mitad (05/06/2024)", DAY, "T1_moderada", None),
    ("caida_dia", "Caída simulada del 20 % todo el día (05/06/2024)", DAY, "T2_moderada", 0),
    ("hora_duplicada", "Error de datos simulado: una hora duplicada (05/06/2024)", DAY, "T4_duplicada", None),
    ("incidente_real", "Incidente real: suspensión del 28/08/2023", pd.Timestamp("2023-08-28"), None, 0),
]


def load_W():
    df, _ = D.load_all(ROOT / "data")
    return D.line_series(df, "LINEA A")


def fit_month(W, day):
    """Los modelos que la evaluación rolling usó para el mes de ``day`` (ajuste con datos < inicio del mes)."""
    hist_end = day.to_period("M").to_timestamp() - pd.Timedelta(days=1)
    hist = W.loc[TRAIN_START:hist_end]
    ae = AEContextDetector(agg="top2", seed=0).fit(hist, W)
    pr = ProfileContextDetector(agg="peak_shift").fit(hist, W)
    return ae, pr


def explain_day(ae, pr, X, W, thr_ae, thr_pr):
    """Detalle hora a hora de una fila X (1 día)."""
    Y = ae._deleveled(X, W)
    Ys = (Y.to_numpy() - ae.mu_) / ae.sd_
    rec = ae.m_.predict(np.hstack([Ys, ae._cal(X.index)]), verbose=0)
    z = (Ys - rec) * ae.sd_ / ae.sig_
    from metro_anomaly.detectors import level_series
    lvl = level_series(W, X.index, ae.exclude_season).to_numpy()[:, None] if hasattr(ae, "exclude_season") else \
        level_series(W, X.index).to_numpy()[:, None]
    esp_ae = np.expm1(rec * ae.sd_ + ae.mu_ + lvl)
    sp = pr.score(X, W)
    esp_pr = np.expm1(pr.expected_log(X.index, W)).to_numpy()
    zs = pd.DataFrame(z, index=X.index, columns=D.SCORED_HOURS)
    s_ae = float(aggregate(zs, "top2").iloc[0])
    order = np.argsort(-np.abs(z[0]))[:2]
    horas = []
    for i, c in enumerate(D.SCORED_HOURS):
        horas.append({"hora": int(c[1:]), "observado": float(X[c].iloc[0]), "entrada": round(float(Ys[0, i]), 4),
                      "reconstruccion": round(float(rec[0, i]), 4), "z_ae": round(float(z[0, i]), 3),
                      "top2": bool(i in order), "esperado_ae": round(float(esp_ae[0, i]), 1),
                      "esperado_perfil": round(float(esp_pr[0, i]), 1), "z_perfil": round(float(sp.z.iloc[0, i]), 3)})
    s_pr = float(sp.day.iloc[0])
    return horas, {"score_ae": round(s_ae, 3), "umbral_ae": round(thr_ae, 4), "alerta_ae": s_ae > thr_ae,
                   "horas_top2": [int(D.SCORED_HOURS[j][1:]) for j in sorted(order)],
                   "score_perfil": round(s_pr, 3), "umbral_perfil": round(thr_pr, 4), "alerta_perfil": s_pr > thr_pr}


def main():
    W = load_W()
    ae1 = json.loads((OUT / "ae1.json").read_text(encoding="utf-8"))["configs"]
    thr_ae, thr_pr = ae1["ae_contextual/top2"]["umbral_dev"], ae1["perfil_contextual/peak_shift"]["umbral_dev"]
    models = {}
    ej_rows, ej_res = [], []
    for k, (eid, titulo, day, scen, seed) in enumerate(EJEMPLOS):
        mkey = str(day.to_period("M"))
        if mkey not in models:
            models[mkey] = fit_month(W, day)
        ae, pr = models[mkey]
        X = W.loc[[day], D.SCORED_HOURS]
        desc = "Dato real"
        if scen:
            s = seed if seed is not None else 0
            # Para que el ejemplo sea legible, se busca una semilla cuya alteración caiga en horas del día comercial.
            for s_try in ([seed] if seed is not None else range(50)):
                Xi, mask, _ = inject(X, scen, s_try, context=W)
                hrs = [int(c[1:]) for c in D.SCORED_HOURS if mask[c].iloc[0]]
                if seed is not None or (scen == "T1_moderada" and len(hrs) == 2 and 11 <= hrs[0] <= 14) or \
                        (scen == "T4_duplicada" and hrs and 7 <= hrs[0] <= 9):
                    s = s_try
                    break
            X = Xi
            desc = f"Simulación ({scen}, horas alteradas: {', '.join(str(h) for h in hrs)})"
        horas, res = explain_day(ae, pr, X, W, thr_ae, thr_pr)
        for h in horas:
            ej_rows.append({"ejemplo": eid, **h})
        ej_res.append({"orden": k + 1, "ejemplo": eid, "titulo": titulo, "tipo_dato": desc, **res})
        print(eid, res, flush=True)

    # Espacio latente (modelo del último mes evaluado) para todos los días reales completos 2022-02…2024-06
    ae, pr = models["2024-06"]
    import tensorflow as tf
    enc = tf.keras.models.Model(ae.m_.inputs, ae.m_.layers[3].output)  # capas: Input, 16, 8, 4 -> cuello de botella
    days = W.loc["2022-02-01":"2024-06-30", D.SCORED_HOURS]
    days = days[days.notna().all(axis=1)]
    Y = ae._deleveled(days, W)
    ok = Y.notna().all(axis=1)
    Y, days = Y[ok], days[ok]
    Ys = (Y.to_numpy() - ae.mu_) / ae.sd_
    Zl = enc.predict(np.hstack([Ys, ae._cal(days.index)]), verbose=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0).fit(Zl)
    P = pca.transform(Zl)
    rec = ae.m_.predict(np.hstack([Ys, ae._cal(days.index)]), verbose=0)
    zz = pd.DataFrame((Ys - rec) * ae.sd_ / ae.sig_, index=days.index, columns=D.SCORED_HOURS)
    sc = aggregate(zz, "top2")
    lat = [{"fecha": str(d.date()), "tipo_dia": t, "pc1": round(float(a), 4), "pc2": round(float(b), 4),
            "score_rel": round(float(s) / thr_ae, 4), "nota": event_note(d) or "", "periodo": "ajuste" if d < pd.Timestamp("2024-06-01") else "evaluado"}
           for d, t, (a, b), s in zip(days.index, day_type(days.index), P, sc)]
    latent_meta = {"varianza_pc": [round(float(v), 3) for v in pca.explained_variance_ratio_], "dim_cuello": int(Zl.shape[1])}

    # Distribución de scores (de las corridas de AE.1) relativa al umbral de cada modelo
    dist = []
    for cfg, fname, thr in (("ae_contextual/top2", "ae_contextual__top2__s0.pkl", thr_ae),
                            ("perfil_contextual/peak_shift", "perfil_contextual__peak_shift__s0.pkl", thr_pr)):
        with open(RUNS / "dev" / fname, "rb") as fh:
            rd = pickle.load(fh)
        with open(RUNS / "holdout" / fname, "rb") as fh:
            rh = pickle.load(fh)
        for v in rd.real["score"].dropna():
            dist.append({"config": cfg, "grupo": "Días reales 2022–2023 (fijan el umbral)", "rel": round(float(v) / thr, 4)})
        for v in rh.real["score"].dropna():
            dist.append({"config": cfg, "grupo": "Días reales del periodo evaluado", "rel": round(float(v) / thr, 4)})
        inj = rh.injected[rh.injected["aplicable"] & rh.injected["score"].notna()]
        for sev, g in inj.groupby("severidad"):
            for v in g["score"]:
                dist.append({"config": cfg, "grupo": f"Anomalías simuladas {sev}s", "rel": round(float(v) / thr, 4)})
        # Mapa |z| por hora del autoencoder en el periodo evaluado
        if cfg.startswith("ae"):
            zr = rh.z_real
            mapa = [{"fecha": str(d.date()), "hora": int(c[1:]), "z": round(float(abs(zr.at[d, c])), 2),
                     "alerta": bool(rh.real.at[d, "score"] > thr)}
                    for d in zr.index for c in D.SCORED_HOURS if pd.notna(zr.at[d, c])]
    out = {"ejemplos": ej_res, "ejemplos_horas": ej_rows, "latente": lat, "latente_meta": latent_meta,
           "distribucion": dist, "mapa": mapa,
           "arquitectura": {"entrada": 19 + ae._cal(days.index[:1]).shape[1], "capas": [16, 8, 4, 8, 16], "salida": 19,
                            "rasgos_calendario": int(ae._cal(days.index[:1]).shape[1])}}
    (OUT / "explicar.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("ok", {k: len(v) if isinstance(v, list) else v for k, v in out.items()})


if __name__ == "__main__":
    main()
