"""Diagnóstico EXPLORATORIO (motivó P1; no pre-registrado): autocorrelación de los residuos (z) del perfil contextual, Línea A, holdout 2023-07..2024-06 (rolling mensual causal). Uso: PYTHONPATH=src python scripts/investigacion_pronostico/diagnostico_residuos.py"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from scipy import stats

from metro_anomaly import data as D
from metro_anomaly.data import SCORED_HOURS
from metro_anomaly.detectors import ProfileContextDetector

df, _ = D.load_all("data"); W = D.line_series(df, "LINEA A")[SCORED_HOURS]
zs = []
for m in pd.period_range("2023-07", "2024-06", freq="M"):
    s, e = m.start_time, m.end_time.normalize()
    det = ProfileContextDetector(agg="peak_shift").fit(W.loc["2022-01-01": s - pd.Timedelta(days=1)], W.loc[: s - pd.Timedelta(days=1)])
    zs.append(det._z(W.loc[s:e], W))
Z = pd.concat(zs).dropna(how="any")
Z = Z.reindex(pd.date_range(Z.index.min(), Z.index.max()))
def acf(x, k):
    x = pd.Series(x); return x.autocorr(k)
def ljung_box(x, K):
    x = pd.Series(x).dropna(); n = len(x)
    q = n * (n + 2) * sum(acf(x, k) ** 2 / (n - k) for k in range(1, K + 1)); return q, 1 - stats.chi2.cdf(q, K)
mz = Z.mean(axis=1)  # desviación media del día
print("días con 19 horas:", int(mz.notna().sum()))
print("Desviación media diaria (mean z):  ACF lag1=%.3f lag2=%.3f lag7=%.3f" % (acf(mz,1), acf(mz,2), acf(mz,7)))
q, p = ljung_box(mz, 14); print("  Ljung-Box(14): Q=%.1f p=%.2g" % (q, p))
c = mz - mz.median()
print("Cuadrados (efecto ARCH): ACF lag1=%.3f lag7=%.3f" % (acf(c**2,1), acf(c**2,7)))
q, p = ljung_box(c**2, 14); print("  Ljung-Box(14) sobre cuadrados: Q=%.1f p=%.2g" % (q, p))
h = Z.to_numpy().ravel()  # serie horaria concatenada (dentro del día)
within = np.mean([pd.Series(r).autocorr(1) for r in Z.dropna().to_numpy()])
print("z horario, ACF lag1 promedio dentro del día=%.3f" % within)
a = c.abs()
print("|desviación| (ARCH robusto): ACF lag1=%.3f lag7=%.3f" % (acf(a,1), acf(a,7)))
q, p = ljung_box(a, 14); print("  Ljung-Box(14) sobre |.|: Q=%.1f p=%.2g" % (q, p))
r = a.rank(); print("  Spearman lag1=%.3f" % acf(r,1))
print("máx |mean z| =%.1f, p99=%.1f" % (a.max(), a.quantile(.99)))
print("Spearman signo mean z lag1=%.3f lag7=%.3f" % (acf(mz.rank(),1), acf(mz.rank(),7)))
print("Spearman |.| lag7=%.3f" % acf(r,7))
