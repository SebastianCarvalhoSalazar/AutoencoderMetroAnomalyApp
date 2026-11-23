# Ficha del modelo — `perfil_contextual` con agregación `peak_shift`

Artefacto: `artifacts/modelo.json` (JSON con tablas, umbral y procedencia; sin pickle). Construcción: `python scripts/build_artifact.py`.

## Qué hace
Para cada día de la Línea A, estima la afluencia esperada por hora (04–22) a partir de:
1. el **tipo de día**: hábil, sábado, domingo/festivo, especial (24/12, 31/12), hábil en temporada, sábado en temporada;
2. un **perfil robusto** (mediana y MAD de log1p(conteo) − nivel) por tipo de día y hora, con una ventana de 365 días;
3. un **nivel causal**: mediana de log(Σ horas) de los días hábiles y sábados no festivos fuera de temporada de [t−28, t−1], con respaldo a 56/91 días.

Con eso calcula **z por hora** = (log1p(obs) − esperado) / sigma, y un **score diario** = max(pico estandarizado, desplazamiento estandarizado):
- pico = (max|z| − mediana) / MAD;
- desplazamiento = |media z| / MAD;
- las escalas se estiman en la historia de entrenamiento.

Hay **alerta** si el score supera **7,388**, el cuantil 95 % de los scores fuera de muestra de DEV (presupuesto de alertas del 5 %).

## Uso previsto
Apoyo a la revisión **retrospectiva o diaria** de la afluencia de la Línea A: señala días y horas con desviaciones respecto de lo esperado para su contexto de calendario y las explica (horas, dirección, magnitud, anotación de calendario).

## Usos no previstos
- Otras líneas (no calibrado; la app las rechaza).
- Decisiones automáticas sin revisión humana.
- Inferir causas: el modelo detecta desviaciones, no explica por qué ocurren.
- Detección intra-día en tiempo real: el diseño es causal y podría extenderse, pero **no se evaluó**.

## Evaluación (protocolo pre-registrado; detalle en `docs/validacion/F4.md`)
HOLDOUT 2023-07-01…2024-06-30, umbral congelado en DEV, IC 95 % por bootstrap semanal:

| Métrica | Valor | Meta | |
|---|---|---|---|
| Tasa de alertas en días reales | 2,2 % [0,8–3,6] | 2–10 % | ✔ |
| Recall anomalías severas (sintéticas) | 0,894 [0,867–0,920] | ≥ 0,95 | ✘ |
| Recall anomalías moderadas | 0,506 [0,462–0,549] | ≥ 0,80 | ✘ |
| Alertas en festivos | 11,1 % (2/18) | ≤ 10 % | ✘ |
| Perfil de festivo en día hábil | 1,00 | ≥ 0,95 | ✔ |
| Frente al autoencoder original (recall moderado) | +0,207 | > 0 | ✔ |
| Transferencia 2019 (tasa de alertas) | 7,5 % | ≤ 15 % | ✔ |
| Eventos reales prospectivos | 4/4 | ≥ 2/3 | ✔ |

## Limitaciones conocidas
- **Sensibilidad moderada (~50 %)**. Recall en el holdout por escenario (`reports/holdout_metricas.csv`):

  | Escenario | Severa | Moderada | Leve |
  |---|---|---|---|
  | T1 caída de 1–3 horas (×0,2 / ×0,5 / ×0,7) | 0,986 | 0,745 | 0,297 |
  | T2 todo el día (×0,6 / ×0,8 / ×0,9) | 0,920 | 0,522 | 0,049 |
  | T3 aumento de 2–3 horas de la tarde (×2 / ×1,5 / ×1,25) | **0,681** | **0,253** | 0,025 |

  T4: hora en cero 1,00, desfase 0,948, hora duplicada 0,761. T5: 1,00. Localización de la hora: 0,983.

  **Los aumentos por evento (T3) son el punto más débil.** En horas pico, una hora aislada debe desviarse alrededor de un ±50 % para salir de la banda.
- Navidad y Año Nuevo generan falsos positivos (comparten tipo de día con los domingos).
- **Cambios de régimen** (por ejemplo, COVID 2020): alertas casi permanentes durante ~3 meses hasta que el perfil se readapta.
- El umbral de DEV resultó conservador en el holdout (2,2 % frente al 5 % previsto).
- La validación primaria usa **anomalías sintéticas**; no hay incidentes operativos verificados en 2022–2024 para validar la utilidad real.

## Mantenimiento recomendado
- Reconstruir el artefacto al incorporar datos nuevos (`scripts/build_artifact.py`) y volver a ejecutar `scripts/reproduce.py` para revisar las métricas.
- Mejoras candidatas **no evaluadas**: tipo de día propio para Navidad y Año Nuevo; recalibración periódica del umbral; contexto multilínea para distinguir eventos sistémicos de locales; retroalimentación de los operadores para construir una verdad de terreno real.
