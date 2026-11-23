"""F6.1 — Reproduce el pipeline completo: calidad de datos → etiquetas → DEV → selección → holdout → externo → artefacto.

Uso:
  python scripts/reproduce.py                 # sin TensorFlow (omite los detectores AE)
  python scripts/reproduce.py --research      # incluye AE (requiere requirements-research.txt)
  python scripts/reproduce.py --workers 4
Sin --research, las comparaciones contra el AE original (D5b) quedan como "n/d"; la selección no cambia,
porque los AE no son elegibles (restricción de festivos o menor recall; ver reports/seleccion_dev.md).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BASE = [("naive_semanal", a) for a in ("max", "top2")] + \
       [(d, a) for d in ("perfil_robusto", "regresion_calendario", "perfil_contextual", "regresion_contextual")
        for a in ("max", "top2", "peak_shift")] + [("isolation_forest", "max")]
RESEARCH = [("ae_original", "max", s) for s in (0, 1, 2)] + [("ae_contextual", "max", s) for s in (0, 1, 2)] + [("ae_contextual", "top2", 0)]
EXTERNAL = ["perfil_robusto/max", "regresion_contextual/max", "naive_semanal/max"]


def _job(args):
    period, det, agg, seed = args
    from run_eval import run
    f, res = run(period, det, agg, seed)
    return f"{period}/{det}/{agg}/s{seed} ({res.fit_seconds:.0f}s)"


def _runs(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(_job, jobs):
            print("  ✔", msg, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    t0 = time.time()
    step = lambda m: print(f"\n== {m} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    step("Calidad de datos y etiquetas")
    importlib.import_module("data_quality").main()
    importlib.import_module("label_quality").main()
    cfgs = [(d, ag, 0) for d, ag in BASE] + (RESEARCH if a.research else [])
    step(f"DEV: {len(cfgs)} configuraciones")
    _runs([("dev", d, ag, s) for d, ag, s in cfgs], a.workers)
    step("Selección")
    importlib.import_module("select_model").main()
    step("HOLDOUT (modelo y umbrales congelados)")
    _runs([("holdout", d, ag, s) for d, ag, s in cfgs], a.workers)
    importlib.import_module("evaluate_holdout").main()
    step("Externo (2019, COVID)")
    import yaml
    sel = yaml.safe_load(open(ROOT / "configs" / "seleccion.yaml"))
    ext = sorted({f"{sel['detector']}/{sel['agg']}", *EXTERNAL} | ({"ae_original/max"} if a.research else set()))
    _runs([(p, *c.split("/"), 0) for p in ("ext2019", "covid") for c in ext], a.workers)
    importlib.import_module("evaluate_external").main()
    step("Artefacto")
    importlib.import_module("build_artifact").main()
    step("Fin")


if __name__ == "__main__":
    main()
