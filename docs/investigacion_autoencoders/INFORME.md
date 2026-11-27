# Informe de la línea de investigación: usos del autoencoder

Rama `claude/investigacion-autoencoders` · 2026-11-26 · Plan con hipótesis pre-registradas: `PLAN.md` · Registro cronológico: `docs/BITACORA.md` · Visualizaciones: tablero "Autoencoders en el Metro".

## Veredicto por hipótesis
| ID | Hipótesis | Resultado (periodo de evaluación 2023-07…2024-06 salvo indicación) | Veredicto |
|---|---|---|---|
| AE.1 | Punto de partida reproducible | Scores idénticos a la fase 2 en las 4 configuraciones. `ae_contextual/top2` empata con el perfil vigente en recall moderado (0,522 frente a 0,506; diferencia +0,016 [−0,038; 0,069]) con el doble de alertas (4,9 % frente a 2,2 %) | Reproducido |
| A1 | Perfil + AE con el mismo 5 % de alertas | Errores de datos (T4) +0,058 [0,027; 0,094], pero recall moderado −0,054 [−0,102; −0,006] | **No cumple** |
| A2 | AE de toda la red frente a la coherencia de red | A −0,141, B −0,247 y 1 −0,117 (IC < 0); solo T-A +0,126; más alertas en festivos | **No cumple** |
| A3 | Completar horas faltantes de los cables | Las horas "faltantes" (04–07 los domingos y festivos) son horas sin servicio según el horario oficial | **Descartada sin ejecutar** (premisa falsa) |
| A4 | El espacio latente separa días con eventos verificados | AUC 0,877 [0,712; 0,985] con 9 eventos; separa elecciones y cierres anticipados, no interrupciones parciales | **Cumple** (descriptiva, con cautela) |
| A5 | Aviso temprano de cambio de régimen (COVID 2020) | Mismo día que el monitor actual (19/03/2020); 1 episodio en 2022–2024 | **No cumple** |
| A6 | Datos por estación cada 15 min | Sin datos | Bloqueada |

## Qué aprendimos
1. **El AE y el perfil ven cosas distintas.** El AE detecta mejor las rupturas de forma: caídas de pocas horas (0,975 frente a 0,745), aumentos por evento y horas duplicadas. Casi no ve los cambios de nivel: caídas de todo el día (0,151 frente a 0,522) o un día hábil con forma de festivo (0,058 frente a 1,000), porque trabaja sobre el perfil sin nivel.
2. **Con una sola cuota de alertas, esa complementariedad no se convierte en mejora neta** (A1): lo que se gana en un tipo se pierde en otro.
3. **Un día muy anómalo contamina la reconstrucción de las horas normales.** El 28/08/2023, el AE alerta pero señala las 6 y las 19 h en lugar de las 18 y las 19 h. Por eso localiza peor la hora (0,935 frente a 0,983).
4. **El AE multilínea no supera a la regla simple de coherencia de red** (mediana de las otras líneas), salvo en el tranvía. Ese resultado secundario merece una hipótesis propia con datos nuevos.
5. **El espacio latente sí resume bien el tipo de día** y aísla las jornadas sin cobro y los cierres: sirve para **explicar** el comportamiento del sistema, no para alertar.
6. **Hallazgo útil fuera del AE:** las horas 04–07 de los cables los domingos y festivos no son datos perdidos sino horas sin servicio. Evaluar cada día con las horas que la línea opera recuperaría la mayoría de los 690 línea-días "no evaluables" de la plataforma (recomendación, sin implementar).

## Decisiones
- **Ningún modelo de alerta cambia.** El perfil contextual sigue en todas las líneas y la coherencia de red en A, B y 1.
- **Lo único corroborado (A4) es descriptivo.** Se integra como explicación: una página de investigación en la app, en esta rama, que muestra cómo funcionan los modelos en uso y qué aprendió el AE. No alerta ni sustituye nada.
- **No se probaron variantes a la vista de los resultados** (otras reglas de combinación, otras arquitecturas). Cualquiera debe pre-registrarse y evaluarse con datos posteriores a junio de 2024.

## Limitaciones
- Las anomalías de la evaluación son **simuladas**. Los únicos eventos reales verificados de la Línea A son 9 días (A4), y las interrupciones parciales no dejan señal en datos línea × hora.
- El periodo de evaluación ya se había inspeccionado en fases anteriores para la Línea A.
- Una sola semilla por configuración en A1–A5. En la fase 2, la variación entre semillas del AE fue pequeña (`reports/holdout.md`), pero no se re-midió aquí.
