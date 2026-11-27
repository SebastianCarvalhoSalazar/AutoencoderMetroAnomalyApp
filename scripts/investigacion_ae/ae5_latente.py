"""AE.5 — Hipótesis A4 (descriptiva): ¿el espacio latente del autoencoder separa los días con eventos verificados?

Diseño fijado antes de calcular (docs/BITACORA.md): distancia de las 4 coordenadas del cuello de cada día a la mediana
de los días de DEV de su mismo tipo; positivos = días de la Línea A en data/eventos_reales.csv (2022-02…2024-06);
AUC con IC por bootstrap; referencia: AUC del score del perfil contextual. Criterio: AUC ≥ 0,85 (solo descriptivo).
Salida: reports/investigacion_ae/ae5.json y ae5.md
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from metro_anomaly import data as D
from metro_anomaly.calendar import day_type
from metro_anomaly.detectors_nn import AEContextDetector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "investigacion_ae"
RUNS = OUT / "runs"


def main():
    df, _ = D.load_all(ROOT / "data")
    W = D.line_series(df, "LINEA A")
    ae = AEContextDetector(agg="top2", seed=0).fit(W.loc["2022-01-01":"2024-05-31"], W)  # modelo de junio de 2024
    import tensorflow as tf
    enc = tf.keras.models.Model(ae.m_.inputs, ae.m_.layers[3].output)
    days = W.loc["2022-02-01":"2024-06-30", D.SCORED_HOURS]
    days = days[days.notna().all(axis=1)]
    Y = ae._deleveled(days, W)
    ok = Y.notna().all(axis=1)
    Y, days = Y[ok], days[ok]
    Z = enc.predict(np.hstack([(Y.to_numpy() - ae.mu_) / ae.sd_, ae._cal(days.index)]), verbose=0)
    tipos = pd.Series(day_type(days.index), index=days.index)
    dev = (days.index >= "2022-04-01") & (days.index <= "2023-06-30")
    cent = {t: np.median(Z[dev & (tipos == t).to_numpy()], 0) for t in tipos.unique() if (dev & (tipos == t).to_numpy()).sum() >= 3}
    dist = np.array([np.linalg.norm(z - cent.get(t, np.median(Z[dev], 0))) for z, t in zip(Z, tipos)])
    ev = pd.read_csv(ROOT / "data" / "eventos_reales.csv")
    pos = set()
    for r in ev.itertuples():
        a = pd.Timestamp(r.fecha_inicio); b = pd.Timestamp(r.fecha_fin) if isinstance(r.fecha_fin, str) and r.fecha_fin else a
        rng = pd.date_range(a, b) if r.id.startswith(("EV", "IN", "MED")) else [a, b]  # OP: solo 24 y 31 de diciembre
        pos |= {d for d in rng if d in days.index}
    y = np.array([d in pos for d in days.index]).astype(int)
    # referencia: score del perfil contextual (corridas rolling de AE.1, DEV + holdout)
    sp = pd.concat([pickle.load(open(RUNS / per / "perfil_contextual__peak_shift__s0.pkl", "rb")).real["score"] for per in ("dev", "holdout")])
    sp = sp[~sp.index.duplicated()].reindex(days.index)
    mask_p = sp.notna().to_numpy()
    rng = np.random.default_rng(0)

    def auc_ci(score, yy):
        a = roc_auc_score(yy, score)
        idx_p, idx_n = np.where(yy == 1)[0], np.where(yy == 0)[0]
        bs = [roc_auc_score(np.r_[np.ones(len(idx_p)), np.zeros(len(idx_n))],
                            np.r_[score[rng.choice(idx_p, len(idx_p))], score[rng.choice(idx_n, len(idx_n))]]) for _ in range(2000)]
        return round(float(a), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)
    a_lat = auc_ci(dist, y)
    a_pr = auc_ci(sp.to_numpy()[mask_p], y[mask_p])
    rows = [{"fecha": str(d.date()), "tipo_dia": t, "distancia": round(float(x), 4), "evento": bool(p)} for d, t, x, p in zip(days.index, tipos, dist, y)]
    out = {"n_positivos": int(y.sum()), "n_negativos": int((1 - y).sum()), "positivos": sorted(str(d.date()) for d in pos),
           "auc_latente": {"auc": a_lat[0], "ic_inf": a_lat[1], "ic_sup": a_lat[2]},
           "auc_perfil_referencia": {"auc": a_pr[0], "ic_inf": a_pr[1], "ic_sup": a_pr[2], "n_dias": int(mask_p.sum())},
           "criterio": {"exigido": "AUC ≥ 0,85", "cumple": bool(a_lat[0] >= 0.85)},
           "rango_eventos": [{"fecha": r["fecha"], "distancia": r["distancia"], "percentil": round(float((dist <= r["distancia"]).mean()), 3)} for r in rows if r["evento"]],
           "dias": rows}
    (OUT / "ae5.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    md = ["# AE.5 — Hipótesis A4 (descriptiva): espacio latente y días con eventos (generado por `scripts/investigacion_ae/ae5_latente.py`)", "",
          f"Positivos (eventos verificados de la Línea A): {out['n_positivos']} · negativos: {out['n_negativos']}", "",
          f"- AUC de la distancia latente: {out['auc_latente']} → {'cumple' if out['criterio']['cumple'] else 'no cumple'} (exigido ≥ 0,85)",
          f"- Referencia, score del perfil contextual: {out['auc_perfil_referencia']}", "",
          pd.DataFrame(out["rango_eventos"]).to_markdown(index=False)]
    (OUT / "ae5.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
