# Línea de investigación: usos del autoencoder en analítica de afluencia

Rama: `claude/investigacion-autoencoders`, creada desde `claude/repository-technical-audit-89x93t` (commit `6fbdbaf`).
Estado: **planificación**. No se ha ejecutado ningún experimento de esta línea.

## Punto de partida (evidencia ya existente)
| Hallazgo | Fuente |
|---|---|
| El autoencoder original (sin contexto) queda por detrás: recall moderado 0,304 frente a 0,506 del perfil contextual (diferencia −0,207 [−0,251; −0,160]) | `reports/holdout.md` |
| El autoencoder con contexto de calendario (`ae_contextual/top2`) **empata** con el perfil contextual en recall moderado (0,522 frente a 0,506; diferencia −0,016 [−0,069; 0,038]), pero alerta en el doble de días (4,9 % frente a 2,2 %) | `reports/holdout.md` |
| `ae_contextual` es mejor en errores de datos: hora duplicada (0,937 frente a 0,761) y perfil desfasado (1,00 frente a 0,948) | `reports/holdout.md` |
| Con protocolo sin fuga, el AE original logra un AUC de 0,855 ± 0,011; con día de la semana como contexto, 0,952 | `audit/AUDITORIA.md` |
| Código disponible: `src/metro_anomaly/detectors_nn.py` (`AEOriginalDetector`, `AEContextDetector`), pruebas de causalidad y determinismo en el job `research` del CI | `tests/unit/test_detectors_nn.py` |

## Protocolo (igual al del proyecto; obligatorio)
- **Particiones:** selección solo en DEV (2022-01…2023-06) y evaluación en el holdout 2023-07…2024-06.
  - Advertencia: el holdout ya se inspeccionó para la Línea A y para el modelado multilínea.
  - Validación adicional: 2019 (transferencia) y, cuando existan, datos posteriores a junio de 2024 (prospectivo).
- **Inyecciones:** el mismo catálogo (`injection.py`, T1–T5). Comparación con el **mismo presupuesto de alertas** (5 % de días reales) y diferencia pareada con IC por bootstrap semanal.
- **Línea base obligatoria:** el modelo vigente de la tarea (perfil contextual, coherencia de red o pronóstico híbrido), además de las líneas base simples.
- **Reproducibilidad:** causalidad (D8) y determinismo (D9) verificados por prueba; semillas fijas; TensorFlow solo en `requirements-research.txt`.
- **Reporte:** se informan los resultados negativos. Una hipótesis que no cumple su criterio se registra como descartada.

## Hipótesis pre-registradas
| ID | Hipótesis | Por qué podría funcionar | Criterio de éxito (pre-registrado) |
|---|---|---|---|
| A1 | **Detector complementario de errores de captura:** la unión del perfil contextual y `ae_contextual`, repartiendo el 5 % de presupuesto, mejora la detección de errores de datos sin perder detección de demanda | El AE ya fue mejor en T4 (duplicada, desfase) | Recall T4 medio ≥ +0,05 frente al perfil solo (IC > 0) y recall moderado T1–T3 sin pérdida (IC del cambio ∋ 0 o > 0) |
| A2 | **Autoencoder multilínea:** reconstruir la matriz del día (11 líneas × 19 horas) detecta anomalías locales igual o mejor que el detector de coherencia de red (H1) | Aprende de forma conjunta la estructura entre líneas y horas | Recall moderado de anomalías locales en la Línea A ≥ coherencia − 0,02 (no inferioridad) **y** mejor en al menos una de B, 1 o T-A; alertas en festivos no peores |
| A3 | **Imputación de horas faltantes** (AE con ruido o enmascarado) para los cables: los 690 línea-días "no evaluables" | La estructura aprendida puede completar horas sin dato | En horas enmascaradas artificialmente, error (WAPE) ≥ 10 % menor que la imputación por perfil; y, al evaluar días imputados, la tasa de alertas no supera el presupuesto. **Las horas imputadas nunca se presentan como dato real** |
| A4 | **Espacio latente para descubrir tipos de día atípicos** no cubiertos por el calendario (eventos, cierres, temporadas) | Agrupa días con forma similar sin etiquetas | Los días de cierre programado, elecciones y festivos verificados (`data/eventos_reales.csv`, `data/cierres_programados.csv`) forman grupos separables: AUC ≥ 0,85 de la distancia latente frente a días normales. Solo descriptivo; no se adopta para alertar sin validación posterior |
| A5 | **Detección de cambio de régimen** (AE sobre ventanas de varios días) con aviso más temprano que el monitor actual | El error de reconstrucción sostenido puede delatar un cambio de nivel | En el estrés COVID 2020 (EV1–EV2), aviso ≥ 3 días antes que el monitor de tasa de alertas, sin advertencias espurias en 2022–2024 (≤ 1 al año) |
| A6 | **Preparación para datos por estación** (cientos de variables cada 15 min) | El AE escala a alta dimensión mejor que los perfiles por celda | Bloqueada hasta tener los datos (`docs/fase3/HOJA_DE_RUTA_DATOS.md`, fuente n.° 1). Se diseñará el experimento, pero no se ejecutará con datos sintéticos presentados como reales |

## Tareas
| ID | Tarea | Estado |
|---|---|---|
| AE.0 | Rama y plan con hipótesis pre-registradas | COMPLETADA (este documento) |
| AE.1 | Reproducir en esta rama los resultados existentes de `ae_contextual` (verificación de punto de partida) | PENDIENTE |
| AE.2 | A1: detector complementario de errores de captura | PENDIENTE |
| AE.3 | A2: autoencoder multilínea frente a la coherencia de red | PENDIENTE |
| AE.4 | A3: imputación de horas faltantes con enmascaramiento | PENDIENTE |
| AE.5 | A4: espacio latente y días atípicos (descriptivo) | PENDIENTE |
| AE.6 | A5: cambio de régimen | PENDIENTE |
| AE.7 | Informe de la línea con veredicto por hipótesis | PENDIENTE |

## Reglas de integración
Nada de esta rama pasa a la plataforma sin:
- cumplir su criterio en el holdout;
- pruebas de causalidad y determinismo;
- un registro en `docs/BITACORA.md`.

Lo que se adopte se integra mediante un PR aparte hacia la rama principal del proyecto.
