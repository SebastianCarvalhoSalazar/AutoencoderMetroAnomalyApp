"""AE.6 — Hipótesis A5: ¿un aviso basado en el autoencoder detecta un cambio de régimen antes que el monitor actual?

Diseño fijado antes de calcular (docs/BITACORA.md):
- Aviso AE: media móvil de 7 días del score diario de ae_contextual/top2 > su p99 en DEV (2022-04…2023-06).
- Aviso del monitor: tasa de alertas del perfil contextual en los últimos 90 días (o los disponibles) > 10 %.
- COVID: corridas causales de 2020 (ajuste desde 2019-01); se compara el primer aviso de cada uno desde EV1 (2020-03-20).
- Falsos avisos del AE: episodios (rachas de días consecutivos con aviso) en 2022-04…2024-06.
Criterio (PLAN.md): AE ≥ 3 días antes que el monitor y ≤ 1 episodio espurio por año.
Salida: reports/investigacion_ae/ae6.json y ae6.md
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "investigacion_ae"
EV1 = pd.Timestamp("2020-03-20")


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def ma7(s: pd.Series) -> pd.Series:
    s = s.asfreq("D")
    return s.rolling(7, min_periods=4).mean()


def episodes(flag: pd.Series) -> list:
    f = flag.fillna(False).astype(bool)
    starts = f & ~f.shift(1, fill_value=False)
    out = []
    for d in f.index[starts]:
        e = d
        while e + pd.Timedelta(days=1) in f.index and f.loc[e + pd.Timedelta(days=1)]:
            e += pd.Timedelta(days=1)
        out.append((d, e))
    return out


def main():
    ae1 = json.loads((OUT / "ae1.json").read_text(encoding="utf-8"))["configs"]
    u_pr = ae1["perfil_contextual/peak_shift"]["umbral_dev"]
    ae_dev = load(OUT / "runs" / "dev" / "ae_contextual__top2__s0.pkl").real["score"]
    ae_hold = load(OUT / "runs" / "holdout" / "ae_contextual__top2__s0.pkl").real["score"]
    m_dev = ma7(ae_dev).loc["2022-04-01":"2023-06-30"]
    thr_ae = float(np.nanquantile(m_dev, 0.99))
    # COVID
    ae_cov = load(OUT / "runs" / "covid" / "ae_contextual__top2__s0.pkl").real["score"]
    pr_cov = load(ROOT / "reports" / "runs" / "covid" / "perfil_contextual__peak_shift__s0.pkl").real["score"]
    m_cov = ma7(ae_cov)
    aviso_ae = m_cov > thr_ae
    al = (pr_cov > u_pr).astype(float).asfreq("D")
    rate = al.rolling(90, min_periods=1).mean()  # el monitor usa los últimos 90 registros o los disponibles
    aviso_mon = rate > 0.10
    first = lambda f: next((d for d in f.index[f.fillna(False).astype(bool)] if d >= EV1), None)
    d_ae, d_mon = first(aviso_ae), first(aviso_mon)
    antes_ev1_ae = [str(a.date()) for a, b in episodes(aviso_ae.loc[:EV1 - pd.Timedelta(days=1)])]
    antes_ev1_mon = [str(a.date()) for a, b in episodes(aviso_mon.loc[:EV1 - pd.Timedelta(days=1)])]
    ventaja = (d_mon - d_ae).days if d_ae is not None and d_mon is not None else None
    # Falsos avisos 2022-04…2024-06
    s_all = pd.concat([ae_dev, ae_hold]).sort_index()
    s_all = s_all[~s_all.index.duplicated()]
    m_all = ma7(s_all).loc["2022-04-01":"2024-06-30"]
    eps = episodes(m_all > thr_ae)
    years = (pd.Timestamp("2024-06-30") - pd.Timestamp("2022-04-01")).days / 365.25
    por_anio = len(eps) / years
    c1 = ventaja is not None and ventaja >= 3
    c2 = por_anio <= 1
    out = {"umbral_ae_ma7_p99_dev": round(thr_ae, 4), "umbral_perfil": u_pr,
           "primer_aviso_ae_desde_ev1": str(d_ae.date()) if d_ae is not None else None,
           "primer_aviso_monitor_desde_ev1": str(d_mon.date()) if d_mon is not None else None,
           "ventaja_dias": ventaja, "avisos_previos_a_ev1": {"ae": antes_ev1_ae, "monitor": antes_ev1_mon},
           "episodios_espurios_2022_2024": [{"inicio": str(a.date()), "fin": str(b.date()), "dias": (b - a).days + 1} for a, b in eps],
           "episodios_por_anio": round(por_anio, 2),
           "criterio": {"ventaja_ge_3_dias": bool(c1), "espurios_le_1_por_anio": bool(c2), "veredicto": "CUMPLE" if c1 and c2 else "NO CUMPLE"},
           "serie_covid": [{"fecha": str(d.date()), "ma7_ae_rel": None if np.isnan(v) else round(float(v) / thr_ae, 4),
                            "tasa90_monitor": None if np.isnan(r) else round(float(r), 4)}
                           for d, v, r in zip(m_cov.loc["2020-01-01":"2020-06-30"].index, m_cov.loc["2020-01-01":"2020-06-30"],
                                              rate.reindex(m_cov.loc["2020-01-01":"2020-06-30"].index))],
           "serie_2022_2024": [{"fecha": str(d.date()), "ma7_ae_rel": None if np.isnan(v) else round(float(v) / thr_ae, 4)} for d, v in m_all.items()]}
    (OUT / "ae6.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    md = ["# AE.6 — Hipótesis A5: cambio de régimen (generado por `scripts/investigacion_ae/ae6_regimen.py`)", "",
          f"Umbral del aviso AE (p99 de la media móvil de 7 días en DEV): {thr_ae:.3f}", "",
          f"- Primer aviso desde EV1 (2020-03-20): AE {out['primer_aviso_ae_desde_ev1']} · monitor {out['primer_aviso_monitor_desde_ev1']} · ventaja {ventaja} días (exigido ≥ 3)",
          f"- Avisos previos a EV1: AE {antes_ev1_ae} · monitor {antes_ev1_mon}",
          f"- Episodios espurios del AE en 2022-04…2024-06: {len(eps)} ({por_anio:.2f} por año; exigido ≤ 1): {out['episodios_espurios_2022_2024']}",
          f"- **Veredicto: {out['criterio']['veredicto']}**"]
    (OUT / "ae6.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
