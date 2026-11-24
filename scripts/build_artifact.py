"""F4.6 — Construye artifacts/modelo.json con la configuración congelada (configs/seleccion.yaml),
ajustada con toda la historia disponible hasta el último día de datos (ventana de 365 días)."""
from pathlib import Path

import pandas as pd
import yaml

from metro_anomaly import data as D
from metro_anomaly.artifacts import git_commit, save_profile, sha256_file
from metro_anomaly.detectors import REGISTRY
from metro_anomaly.evaluation import budget_threshold, rolling_evaluate
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    sel = yaml.safe_load(open(ROOT / "configs" / "seleccion.yaml"))
    df, _ = D.load_all(ROOT / "data")
    W = D.line_series(df, "LINEA A")
    end = W.index.max()
    det = REGISTRY[sel["detector"]](agg=sel["agg"]).fit(W.loc["2022-01-01":end], W)
    hold = pd.read_csv(ROOT / "reports" / "holdout_metricas.csv", index_col=0)
    cfg = f"{sel['detector']}/{sel['agg']}"
    keys = ["tasa_alertas", "tasa_alertas_festivos", "recall_severa", "recall_moderada", "recall_leve", "recall_T5_perfil_festivo"]
    # Distribución de referencia: scores fuera de muestra de los días reales de DEV (rolling mensual, sin inyección).
    dev = rolling_evaluate(lambda: REGISTRY[sel["detector"]](agg=sel["agg"]), W, "2022-04-01", "2023-06-30", "2022-01-01", scenarios=[])
    ref = np.sort(dev.real["score"].dropna().to_numpy())
    assert np.isclose(budget_threshold(dev.real["score"]), sel["umbral"]), "el umbral no coincide con la referencia de DEV"
    meta = {
        "reference_scores": {"periodo": "DEV 2022-04-01…2023-06-30 (rolling mensual, días reales, fuera de muestra)",
                             "n": int(len(ref)), "p95": float(np.quantile(ref, 0.95)), "p99": float(np.quantile(ref, 0.99)),
                             "max": float(ref.max()), "valores_ordenados": [round(float(x), 6) for x in ref]},
        "line": "LINEA A",
        "trained_on": [str((end - pd.Timedelta(days=det.window_days - 1)).date()), str(end.date())],
        "threshold_provenance": sel["procedencia_umbral"],
        "data_sha256": {p.name: sha256_file(p) for p in sorted((ROOT / "data").glob("Afluencia_Metro_*.xlsx"))},
        "git_commit_build": git_commit(ROOT),
        "holdout_metrics": {k: round(float(hold.loc[cfg, k]), 4) for k in keys},
        "notes": "Modelo de perfil contextual causal; ver docs/validacion/F4.md y la ficha del modelo (docs/MODEL_CARD.md).",
    }
    out = save_profile(det, sel["umbral"], ROOT / "artifacts" / "modelo.json", meta)
    print(out)


if __name__ == "__main__":
    main()
