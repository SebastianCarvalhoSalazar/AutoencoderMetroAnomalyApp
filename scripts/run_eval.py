"""Ejecuta la evaluación rolling de un detector en un periodo y guarda el resultado.

Uso: python scripts/run_eval.py --period dev --detector perfil_robusto --agg max --seed 0
Periodos:
  dev       ajuste desde 2022-01-01, evaluación 2022-04-01…2023-06-30 (selección de modelo)
  holdout   ajuste desde 2022-01-01, evaluación 2023-07-01…2024-06-30 (una sola vez, con el modelo congelado)
  ext2019   transferencia: ajuste con DEV (2022-01…2023-06), evaluación 2019-03-01…2019-12-31
  covid     causal: ajuste desde 2019-01-01, evaluación 2020-01-01…2021-12-31 (estrés y eventos EV1/EV2)
"""
import argparse
import pickle
from pathlib import Path

from metro_anomaly import data as D
from metro_anomaly.detectors import REGISTRY
from metro_anomaly.evaluation import DEV, HOLDOUT, rolling_evaluate

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {
    "dev": dict(train_start="2022-01-01", eval_start="2022-04-01", eval_end=DEV[1]),
    "holdout": dict(train_start="2022-01-01", eval_start=HOLDOUT[0], eval_end=HOLDOUT[1]),
    "ext2019": dict(train_start="2022-01-01", eval_start="2019-03-01", eval_end="2019-12-31", fixed_train=DEV),
    "covid": dict(train_start="2019-01-01", eval_start="2020-01-01", eval_end="2021-12-31"),
}


def load_W():
    df, _ = D.load_all(ROOT / "data")
    return D.line_series(df, "LINEA A")


def factory_for(name, agg, seed):
    if name in REGISTRY:
        cls = REGISTRY[name]
    else:
        from metro_anomaly.detectors_nn import REGISTRY_NN
        cls = REGISTRY_NN[name]
    kw = {"agg": agg}
    if "seed" in cls.__init__.__code__.co_varnames:
        kw["seed"] = seed
    return lambda: cls(**kw)


def run(period, detector, agg="max", seed=0, out_dir=None, W=None):
    W = load_W() if W is None else W
    res = rolling_evaluate(factory_for(detector, agg, seed), W, seed=seed, **PERIODS[period])
    res.meta = {"period": period, "detector": detector, "agg": agg, "seed": seed, **PERIODS[period]}
    out = Path(out_dir or ROOT / "reports" / "runs" / period)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{detector}__{agg}__s{seed}.pkl"
    with open(f, "wb") as fh:
        pickle.dump(res, fh)
    return f, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=PERIODS)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--agg", default="max")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    f, res = run(a.period, a.detector, a.agg, a.seed)
    print(f, f"fit_s={res.fit_seconds:.1f}", len(res.real), len(res.injected))
