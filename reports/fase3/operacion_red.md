# Simulación cronológica de la operación diaria multilínea (generado por `scripts/fase3/simulacion_operacion_red.py`)

Periodo 2024-06-01 … 2024-06-29 (29 días). Cada día t se evalúa con datos ≤ t y se emite el pronóstico de t+1.

- Línea-días registrados: 319; coinciden con la reproducción histórica (`dias.csv.gz`): **319/319**
- Latencia por día — evaluación de 11 líneas: mediana 8.01 s, p95 8.38 s; pronóstico: mediana 50.83 s, p95 52.23 s
- Alertas del periodo: 2024-06-04 LINEA O, 2024-06-06 LINEA M, 2024-06-08 LINEA 1, 2024-06-12 LINEA T-A, 2024-06-12 LINEA M, 2024-06-17 LINEA 2, 2024-06-25 LINEA J, 2024-06-25 LINEA 1
- Advertencias del monitor: ['LINEA P: cobertura del intervalo del 80 % = 0.63 fuera de [0.7, 0.9] en 29 días; recalibrar.']

## Pronóstico con reajuste diario (operación) frente a reajuste mensual (reproducción del holdout)

Mismos días y franjas; el intervalo del reajuste diario usa el factor conformal del manifiesto.

|           |   wape_diario |   wape_mensual |   cobertura_diario |   cobertura_mensual |
|:----------|--------------:|---------------:|-------------------:|--------------------:|
| LINEA 1   |        0.0397 |         0.04   |              0.819 |               0.82  |
| LINEA 2   |        0.0612 |         0.0615 |              0.856 |               0.84  |
| LINEA A   |        0.0347 |         0.0346 |              0.84  |               0.858 |
| LINEA B   |        0.0524 |         0.0539 |              0.813 |               0.831 |
| LINEA H   |        0.0896 |         0.0902 |              0.867 |               0.853 |
| LINEA J   |        0.0579 |         0.0587 |              0.824 |               0.808 |
| LINEA K   |        0.0471 |         0.0468 |              0.901 |               0.883 |
| LINEA M   |        0.0702 |         0.0697 |              0.845 |               0.832 |
| LINEA O   |        0.0727 |         0.0729 |              0.857 |               0.858 |
| LINEA P   |        0.0919 |         0.276  |              0.644 |               0.677 |
| LINEA T-A |        0.0648 |         0.0669 |              0.797 |               0.813 |
