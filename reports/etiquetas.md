# Calidad de las etiquetas heurísticas (generado por `scripts/label_quality.py`)

Periodo: Línea A 2022-01-01…2024-06-30, horas 04–22. Variantes: `raw_k1.5` = comportamiento del notebook original (IQR sobre valores crudos); `resid_k1.5` = lo que el notebook declaraba (IQR sobre residuos multiplicativos); `resid_k3.0` = criterio más estricto.

```json
{
  "n_dias": 906,
  "prevalencia": {
    "raw_k1.5": 0.249,
    "resid_k1.5": 0.254,
    "resid_k3.0": 0.111
  },
  "kappa": {
    "raw_k1.5 vs resid_k1.5": 0.525,
    "raw_k1.5 vs resid_k3.0": 0.483,
    "resid_k1.5 vs resid_k3.0": 0.539
  },
  "tasa_en_festivos": {
    "raw_k1.5": 0.955,
    "resid_k1.5": 0.955,
    "resid_k3.0": 0.955
  },
  "tasa_en_diciembre": {
    "raw_k1.5": 0.984,
    "resid_k1.5": 0.565,
    "resid_k3.0": 0.323
  },
  "tasa_resto": {
    "raw_k1.5": 0.158,
    "resid_k1.5": 0.195,
    "resid_k3.0": 0.053
  },
  "tasa_por_tipo_dia": {
    "raw_k1.5": {
      "domingo_festivo": 0.413,
      "especial": 1.0,
      "habil": 0.209,
      "sabado": 0.205
    },
    "resid_k1.5": {
      "domingo_festivo": 0.383,
      "especial": 1.0,
      "habil": 0.215,
      "sabado": 0.244
    },
    "resid_k3.0": {
      "domingo_festivo": 0.317,
      "especial": 1.0,
      "habil": 0.056,
      "sabado": 0.079
    }
  }
}
```
