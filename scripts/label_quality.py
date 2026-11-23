"""F3.1 — Calidad de las etiquetas heurísticas (Línea A, 2022-01-01…2024-06-30) -> reports/etiquetas.md"""
import json
from pathlib import Path

from metro_anomaly import data as D
from metro_anomaly.labels import label_quality

ROOT = Path(__file__).resolve().parents[1]


def main():
    df, _ = D.load_all(ROOT / "data")
    W = D.line_series(df, "LINEA A", "2022-01-01", "2024-06-30")
    q = label_quality(W)
    txt = ["# Calidad de las etiquetas heurísticas (generado por `scripts/label_quality.py`)", "",
           "Periodo: Línea A 2022-01-01…2024-06-30, horas 04–22. Variantes: `raw_k1.5` = comportamiento del notebook original "
           "(IQR sobre valores crudos); `resid_k1.5` = lo que el notebook declaraba (IQR sobre residuos multiplicativos); "
           "`resid_k3.0` = criterio más estricto.", "", "```json", json.dumps(q, indent=2, ensure_ascii=False), "```", ""]
    out = ROOT / "reports" / "etiquetas.md"
    out.write_text("\n".join(txt), encoding="utf-8")
    print(json.dumps(q, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
