"""F1.2 — Cada archivo de demo debe corresponder exactamente a un día del dataset con el scaler desplegado,
y su etiqueta (nombre del archivo) debe coincidir con la etiqueta del dataset."""
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


def _match_all():
    df = pd.read_excel(ROOT / "dev/results/full_2019_2024_data_linea_A_outliers.xlsx")
    hours = [c for c in df.columns if isinstance(c, str) and c.endswith(":00:00") and not c.endswith("_outlier")]
    X = df[hours].to_numpy(float)
    sc = joblib.load(ROOT / "dev/results/scaler_77_81_78.pkl")
    rows = []
    for f in sorted((ROOT / "dev/data_test").glob("*.pkl")):
        v = pickle.loads(f.read_bytes())  # archivos del propio repositorio (confiables)
        raw = sc.inverse_transform(v.reshape(1, -1))[0]
        d = np.abs(X - raw).max(axis=1)
        j = int(d.argmin())
        rows.append((f.name, float(d[j]), int("outlier" in f.name), int(df["Any_Outlier_outlier"].iloc[j])))
    return rows


def test_every_demo_file_is_an_exact_dataset_day_with_consistent_label():
    rows = _match_all()
    assert len(rows) > 0
    stale = [r[0] for r in rows if r[1] > 1e-6]
    assert stale == [], f"{len(stale)} archivos no corresponden al scaler desplegado: {stale[:5]}..."
    wrong = [r[0] for r in rows if r[2] != r[3]]
    assert wrong == []
