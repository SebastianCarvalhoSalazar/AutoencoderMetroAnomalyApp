"""F10.1 — Inventario y análisis exploratorio multilínea -> reports/fase3/eda_multilinea.md y .json

Uso: python scripts/eda_multilinea.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metro_anomaly import data as D
from metro_anomaly.calendar import day_type, is_holiday

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fase3"
H = D.SCORED_HOURS


def wide_all(df):
    return {ln: D.line_series(df, ln)[D.HOUR_COLS] for ln in sorted(df["line"].unique())}


def robust_z(W: pd.DataFrame, t, hours, weeks=8):
    """z robusto de cada hora del día t frente a los mismos días de la semana de las `weeks` semanas previas."""
    t = pd.Timestamp(t)
    refs = [t - pd.Timedelta(weeks=k) for k in range(1, weeks + 1)]
    refs = [r for r in refs if r in W.index and not is_holiday([r])[0]]
    R = np.log1p(W.loc[refs, hours])
    med, mad = R.median(), 1.4826 * (R - R.median()).abs().median()
    return ((np.log1p(W.loc[t, hours]) - med) / mad.clip(lower=0.03)).round(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, rep = D.load_all(ROOT / "data")
    lines = pd.read_csv(ROOT / "data" / "lineas.csv").set_index("linea")
    Ws = wide_all(df)
    res = {"lineas": {}}
    # 1. Inventario por línea
    rows = []
    for ln, W in Ws.items():
        obs = W[W.notna().any(axis=1)]
        hours_active = obs.notna().sum(axis=1)
        for y, g in obs.groupby(obs.index.year):
            tot = g.sum(axis=1, min_count=1)
            rows.append({"línea": ln, "modo": lines.loc[ln, "modo"] if ln in lines.index else "?", "año": y, "días": len(g),
                         "horas_con_dato_mediana": int(hours_active.loc[g.index].median()),
                         "entradas_diarias_media": int(tot.mean()), "pct_celdas_nan_04_22": round(100 * g[H].isna().mean().mean(), 1),
                         "pct_ceros_04_22": round(100 * (g[H] == 0).mean().mean(), 2)})
    inv = pd.DataFrame(rows)
    res["inventario"] = inv.to_dict(orient="records")
    # 2. Participación por línea (2023)
    tot23 = {ln: float(W.loc["2023"].sum().sum()) for ln, W in Ws.items()}
    s = sum(tot23.values())
    res["participacion_2023_pct"] = {ln: round(100 * v / s, 2) for ln, v in sorted(tot23.items(), key=lambda x: -x[1])}
    # 3. Correlaciones de la demanda diaria (2022-07…2024-06), en log y desestacionalizada (vs. mismo día de la semana, 8 semanas)
    idx = pd.date_range("2022-07-01", "2024-06-30")
    daily = pd.DataFrame({ln: np.log(W.reindex(idx).sum(axis=1, min_count=10)) for ln, W in Ws.items()})
    main_lines = [ln for ln in daily.columns if daily[ln].notna().mean() > 0.95]
    dl = daily[main_lines]
    res["corr_log_diaria"] = dl.corr().round(2).to_dict()
    resid = dl - dl.rolling("56D", closed="left", min_periods=20).median()
    resid = resid.groupby(resid.index.dayofweek).transform(lambda x: x - x.rolling(8, min_periods=4).median().shift(1))
    rr = resid.dropna()
    res["corr_residuo_diario"] = rr.corr().round(2).to_dict()
    ev = np.linalg.eigvalsh(np.corrcoef(rr.T.values))[::-1]
    res["pca_residuo_varianza_explicada"] = [round(float(x / ev.sum()), 3) for x in ev[:4]]
    # 4. Huecos simultáneos (posibles fallas de recaudo de red): horas en que ≥ 3 líneas habituales no tienen dato
    simult = {}
    for t in pd.date_range("2019-01-01", "2024-06-30"):
        cnt = {}
        for h in H:
            n = 0
            for ln in main_lines:
                W = Ws[ln]
                if t in W.index and W[h].loc[t - pd.Timedelta(days=28): t - pd.Timedelta(days=1)].notna().mean() > 0.9 and np.isnan(W.loc[t, h]):
                    n += 1
            if n >= 3:
                cnt[h] = n
        if cnt:
            simult[str(t.date())] = cnt
    res["huecos_simultaneos_>=3_lineas"] = simult
    # 5. Eventos conocidos: z por línea y hora
    evs = {"IN3_2023-08-28": ("2023-08-28", ["h16", "h17", "h18", "h19", "h20"]),
           "IN2_2023-08-22": ("2023-08-22", ["h06", "h07", "h08", "h09"]),
           "IN1_2023-02-05": ("2023-02-05", ["h17", "h18", "h19", "h20"]),
           "MED1_2023-10-29": ("2023-10-29", ["h08", "h12", "h16", "h17"])}
    res["eventos_z"] = {}
    for k, (d, hs) in evs.items():
        res["eventos_z"][k] = {ln: robust_z(Ws[ln], d, hs).to_dict() for ln in main_lines if pd.Timestamp(d) in Ws[ln].index}
    res["lineas_principales"] = main_lines
    (OUT / "eda_multilinea.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = ["# Inventario y EDA multilínea (generado por `scripts/eda_multilinea.py`)", "",
          f"Archivos: {', '.join(rep.files)} · duplicados entre archivos resueltos: {rep.cross_file_duplicates}", "",
          "## Inventario por línea y año", "", inv.to_markdown(index=False), "",
          "## Participación en las entradas de 2023 (%)", "", pd.Series(res["participacion_2023_pct"]).to_frame("%").to_markdown(), "",
          "## Correlación de la demanda diaria desestacionalizada (2022-07…2024-06)", "", rr.corr().round(2).to_markdown(), "",
          f"Varianza explicada por los primeros componentes principales del residuo: {res['pca_residuo_varianza_explicada']}", "",
          f"## Horas con ≥ 3 líneas sin dato a la vez (posibles fallas de recaudo de red): {len(simult)} días", "",
          "\n".join(f"- {k}: {v}" for k, v in list(simult.items())[:40]), "",
          "## z robusto por línea en eventos conocidos (frente al mismo día de la semana de las 8 semanas previas)", ""]
    for k, v in res["eventos_z"].items():
        md += [f"### {k}", "", pd.DataFrame(v).T.to_markdown(), ""]
    (OUT / "eda_multilinea.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)[:12000])


if __name__ == "__main__":
    main()
