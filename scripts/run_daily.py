"""Procesamiento operativo por lote (fin de día) con registro de auditoría.

Ejemplos:
  python scripts/run_daily.py --fecha 2023-08-28
  python scripts/run_daily.py --archivo nuevo.xlsx                  # evalúa todos los días del archivo
  python scripts/run_daily.py --desde 2023-07-01 --hasta 2024-06-30 --monitor
Códigos de salida: 0 = OK · 2 = error de entrada · 3 = error del modelo/artefacto.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from metro_anomaly import operations as O
from metro_anomaly import service as S

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha"); ap.add_argument("--desde"); ap.add_argument("--hasta"); ap.add_argument("--archivo")
    ap.add_argument("--modelo", default=str(ROOT / "artifacts" / "modelo.json"))
    ap.add_argument("--datos", default=str(ROOT / "data"))
    ap.add_argument("--log", default=str(ROOT / "reports" / "operacion" / "decisiones.jsonl"))
    ap.add_argument("--monitor", action="store_true")
    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
