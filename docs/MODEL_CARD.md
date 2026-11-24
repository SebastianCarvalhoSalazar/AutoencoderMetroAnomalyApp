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

Además, reglas de calidad de datos (ceros sospechosos; valores > 1,5 × máximo histórico de la hora) y el contexto de jornadas electorales (no cobro) clasifican el día como *posible problema de datos* o *efecto de medición conocido*. Hay **alerta** si el score supera **7,388**, el cuantil 95 % de los scores fuera de muestra de DEV (presupuesto de alertas del 5 %).

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
| Eventos reales prospectivos | 4/4 (2 COVID + 2 elecciones; las elecciones son un **efecto de medición** por no cobro) | ≥ 2/3 | ✔ formal |
| Incidentes operacionales verificados (fuentes oficiales, F8) | **1/3** (no detecta operaciones parciales) | — | ✘ |

## Validación de producción (F8)
No anticipa eventos (en el incidente real del 28/08/2023 habría alertado ~30 min después del aviso oficial, con dato horario hipotético). El dato cuenta validaciones de pago, por lo que la gratuidad lo distorsiona. Dictamen: `docs/validacion/DICTAMEN_PRODUCCION.md`.

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
- Mejoras candidatas **no evaluadas**: tipo de día propio para Navidad y Año Nuevo; recalibración periódica del umbral; retroalimentación de los operadores para construir una verdad de terreno real. El contexto multilínea **se evaluó en la fase 3** (sección siguiente).

---

# Modelos de la plataforma multilínea (fase 3)
Evidencia completa en `docs/validacion/F10_fase3_plataforma.md`, `reports/fase3/red.md` y `reports/fase3/pronostico_{dev,holdout}.md`. Artefactos con procedencia en `artifacts/plataforma/manifiesto.json`.

| Modelo | Uso previsto | Cómo funciona | Evidencia (fuera de muestra) | Estado |
|---|---|---|---|---|
| Perfil contextual por línea | Desviación por línea y hora (z) y score diario | El mismo modelo de arriba, ajustado cada mes para cada línea con datos anteriores al mes. Umbral = p95 del score en DEV de cada línea | — | Adoptado en las 11 líneas (detección en 8) |
| **Coherencia de red** (`red.CoherenceDetector`) | Anomalía propia de la línea, descontando lo que mueve a toda la red | z de la línea − (a_h + b_h · mediana de z de las otras 10 líneas en la misma hora), con ajuste Huber por hora y escala MAD. Los ceros de otras líneas se excluyen | Línea A: recall moderado 0,702 → 0,900, diferencia [0,177; 0,216]; menos alertas en festivos. B: +0,052; 1: +0,091; T-A: sin mejora | Adoptado en A, B y 1 |
| Pronóstico del día siguiente (`pronostico.forecast_day`) | Entradas de mañana por línea y hora, con intervalo del 80 % | Media en escala log de un GBM global (calendario, nivel causal, último día comparable por tipo extendido) y del perfil contextual (`combo_v2`; `gbm_v2` en la O y `combo` en la M, según DEV). Banda cuantílica p10–p90 del GBM, recentrada y escalada con un factor conformal | WAPE 11,5–27,6 % menor que el naive en las 11 líneas. Cobertura de la banda calibrada: 0,751–0,831 | Adoptado. La calibración es posterior a ver los resultados (evidencia más débil) |
| Día de alta demanda (H6) | ¿Superará mañana el p95 del último año? | Total diario del GBM base frente al p95 de los 365 días previos | AUC 0,84–0,97 y mejor que el naive en A, 2, J, M, O, P y T-A | Adoptado solo en esas 7 líneas |
| Alcance (H2) | Local / multilínea / generalizada | Fracción de otras líneas con \|z\| > 3 en el mismo sentido y las mismas horas (cortes 0,2 y 0,5) | Exactitud 0,884 (< 0,90) | Solo orientativo |

**Reglas, con prioridad:** efecto de medición (elecciones en metro, tranvía y cable) > ausencia simultánea de registro en ≥ 3 líneas habituales > problema de datos > anomalía > sin anomalía. Los días incompletos en horas que la línea no siempre reporta son "no evaluable".

**Usos no previstos:**
- detección temprana o en tiempo real de incidentes;
- localización de la estación o el tramo;
- diagnóstico de fallas técnicas;
- recomendaciones de capacidad o frecuencia.

Los incidentes parciales IN1 e IN2 no dejan señal en datos línea × hora.

**Operación:**
- `python scripts/run_daily.py --red --fecha AAAA-MM-DD --pronostico --monitor` evalúa las 11 líneas y emite el pronóstico del día siguiente.
- Registra la auditoría en JSON Lines idempotente y monitorea, por línea, la tasa de alertas y la cobertura del intervalo.
- Coincide con la reproducción histórica: 33/33 línea-días en la prueba de integración.
- Diferencia: la operación reajusta el pronóstico cada día, mientras que la reproducción del holdout lo reajustaba cada mes.

**Mantenimiento:**
- reconstruir los artefactos (`python scripts/fase3/build_plataforma.py`, ~12 min) cuando lleguen datos nuevos;
- verificar el determinismo (`scripts/fase3/check_determinism_plataforma.py`);
- recalibrar si el monitor advierte una cobertura fuera de [0,70; 0,90] o una tasa de alertas por línea > 10 %.
