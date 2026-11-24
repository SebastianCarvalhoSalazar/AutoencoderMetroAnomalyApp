"""Procesamiento operativo por lote (fin de día) con registro de auditoría.

Ejemplos:
  python scripts/run_daily.py --fecha 2023-08-28
  python scripts/run_daily.py --archivo nuevo.xlsx                  # evalúa todos los días del archivo
  python scripts/run_daily.py --desde 2023-07-01 --hasta 2024-06-30 --monitor
  python scripts/run_daily.py --red --fecha 2023-08-28 --pronostico --monitor   # 11 líneas + pronóstico del día siguiente
Códigos de salida: 0 = OK · 2 = error de entrada · 3 = error del modelo/artefacto.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from metro_anomaly import data as D
from metro_anomaly import operacion_red as OR
from metro_anomaly import operations as O
from metro_anomaly import plataforma as PL
from metro_anomaly import service as S

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha"); ap.add_argument("--desde"); ap.add_argument("--hasta"); ap.add_argument("--archivo")
    ap.add_argument("--modelo", default=str(ROOT / "artifacts" / "modelo.json"))
    ap.add_argument("--datos", default=str(ROOT / "data"))
    ap.add_argument("--log", default=str(ROOT / "reports" / "operacion" / "decisiones.jsonl"))
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--red", action="store_true", help="modo multilínea (11 líneas) con el manifiesto de la plataforma")
    ap.add_argument("--pronostico", action="store_true", help="(--red) emite el pronóstico del día siguiente al último evaluado")
    ap.add_argument("--manifiesto", default=str(ROOT / "artifacts" / "plataforma" / "manifiesto.json"))
    ap.add_argument("--log-red", default=str(ROOT / "reports" / "operacion" / "decisiones_red.jsonl"))
    ap.add_argument("--log-pronosticos", default=str(ROOT / "reports" / "operacion" / "pronosticos.jsonl"))
    a = ap.parse_args(argv)
    if a.red:
        return main_red(a, ap)
    try:
        hist = S.load_history(a.datos)
        new = None
        if a.archivo:
            p = Path(a.archivo)
            new = S.parse_upload(p.read_bytes(), p.name)
            dates = list(new.index)
        elif a.fecha:
            dates = [pd.Timestamp(a.fecha)]
        elif a.desde and a.hasta:
            dates = list(pd.date_range(a.desde, a.hasta))
        else:
            ap.error("indique --fecha, --archivo o --desde/--hasta")
        summary = O.process(dates, hist, a.modelo, a.log, new)
    except O.ModelError as e:
        print(json.dumps({"error_modelo": str(e)}, ensure_ascii=False)); return O.EXIT_MODEL
    except (S.InputError, ValueError, FileNotFoundError) as e:
        print(json.dumps({"error_entrada": str(e)}, ensure_ascii=False)); return O.EXIT_INPUT
    print(json.dumps(summary, ensure_ascii=False))
    if a.monitor:
        print(json.dumps(O.monitor(a.log), ensure_ascii=False, indent=1))
    return O.EXIT_OK


def main_red(a, ap) -> int:
    try:
        red = PL.load_network(a.datos)
        if a.archivo:
            p = Path(a.archivo)
            new = OR.parse_upload_red(p.read_bytes(), p.name)
            red = OR.merge_red(red, new)
            dates = sorted(set().union(*[set(W.index) for W in new.values()]))
        elif a.fecha:
            dates = [pd.Timestamp(a.fecha)]
        elif a.desde and a.hasta:
            dates = list(pd.date_range(a.desde, a.hasta))
        else:
            ap.error("indique --fecha, --archivo o --desde/--hasta")
        out = {"decisiones": OR.process_red(dates, red, a.manifiesto, a.log_red)}
        if a.pronostico:
            out["pronostico"] = OR.forecast_red(max(dates) + pd.Timedelta(days=1), red, a.manifiesto, a.log_pronosticos)
    except OR.ManifestError as e:
        print(json.dumps({"error_modelo": str(e)}, ensure_ascii=False)); return O.EXIT_MODEL
    except (D.DataValidationError, ValueError, FileNotFoundError) as e:
        print(json.dumps({"error_entrada": str(e)}, ensure_ascii=False)); return O.EXIT_INPUT
    print(json.dumps(out, ensure_ascii=False))
    if a.monitor:
        print(json.dumps(OR.monitor_red(a.log_red, a.log_pronosticos, red), ensure_ascii=False, indent=1))
    return O.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
