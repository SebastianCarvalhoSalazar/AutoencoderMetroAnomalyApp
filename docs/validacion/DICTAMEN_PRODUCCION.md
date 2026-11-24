# Dictamen de preparación para producción — Metro de Medellín, Línea A

Fecha: 2026-11-23 · Evidencia: `docs/validacion/F8_F9_produccion_y_ux.md`, `reports/produccion/validacion_produccion.json`, `reports/ux/`, `docs/validacion/img/ux/`.

## 1. Dictamen
**NO APTO como mecanismo de detección temprana de incidentes operacionales.**

**APTO CON CONDICIONES** como herramienta de apoyo para **revisión retrospectiva diaria** de la afluencia y **control de calidad de los datos** de recaudo, siempre que las alertas se traten como "anomalía estadística a verificar" y nunca como incidente.

| Componente | Estado | Fundamento |
|---|---|---|
| Pipeline de datos | **Listo** (para el formato de export actual) | Validación de esquema, sin imputación, reglas de calidad, sin fuga (0 violaciones), determinista, robusto ante faltantes, retrasos y corrupción |
| Backend / servicio | **Listo para uso interno** | 0,04 s por día; auditoría JSONL con hashes; idempotente; fallos controlados (códigos 2 y 3); estabilidad de decisiones del 99,7 % |
| Modelo | **Insuficiente para el objetivo operacional** | Incumple D2 y D3; sensibilidad a incidentes reales 1/3; no anticipa; no localiza; aumentos y cambios sostenidos poco visibles |
| Frontend / UX | **Listo para piloto interno** | 0 errores de JS o excepciones; UI = backend al 100 %; 0 violaciones de accesibilidad atribuibles a la app; comunicación honesta de las alertas |
| Operación (despliegue) | **No listo** | Sin autenticación, sin orquestación programada, sin canal de notificación, sin integración con la bitácora del Centro de Control ni flujo de reconocimiento de alertas |

## 2. Qué decisiones puede soportar y cuáles no
**Puede soportar** (con revisión humana):
- Priorizar qué días revisar en un análisis retrospectivo de afluencia (≈ 2 % de los días, con horas y magnitud).
- Detectar problemas de calidad del dato de recaudo: horas faltantes, conteos acumulados en la hora siguiente, ceros sospechosos, errores de unidad. Es el aporte con mayor evidencia (artefactos de 2019 identificados).
- Cuantificar después del hecho el impacto en la afluencia de un evento ya conocido (p. ej., 28/08/2023: −32 % en el día, −90 % a las 18–19 h).
- Detectar perfiles incoherentes con el calendario (100 % en simulación) y caídas o picos severos de pocas horas (≈ 99 %).

**No puede justificar:**
- Decisiones preventivas o de despacho, gestión de aglomeraciones ni alertas tempranas: **no anticipa**. Detecta después de que ocurre la desviación; con dato horario hipotético, ~1 h después del inicio. En el incidente real del 28/08/2023 habría alertado ~30 min **después** del aviso oficial del Metro.
- Atribuir causas: la alerta no distingue entre falla técnica, persona en la vía, evento masivo, clima o gratuidad.
- Localizar el tramo o la estación afectada: los datos están agregados por línea.
- Descartar incidentes cuando no hay alerta: 2 de 3 incidentes reales verificados (operación parcial) **no** dejaron señal.
- Vigilar problemas sostenidos: el nivel causal absorbe un escalón de −20 % en pocos días.

## 3. Riesgos y bloqueantes
| # | Tipo | Riesgo | Severidad | Evidencia | Mitigación necesaria |
|---|---|---|---|---|---|
| B1 | **Bloqueante** | No existe verdad de terreno operacional: la utilidad para incidentes no es demostrable | Crítica | §3 del informe F8/F9 | Bitácora del Centro de Control + protocolo de modo sombra (`docs/PROTOCOLO_VALIDACION_OPERACIONAL.md`) |
| B2 | **Bloqueante** | Granularidad insuficiente: el export diario agrega por línea y por hora; las interrupciones parciales son invisibles (IN1, IN2) | Crítica | IN1 max \|z\| 1,3; IN2 max \|z\| 2,6 | Datos por estación, cada 15 min y en flujo casi en tiempo real; salidas y operación de trenes |
| B3 | **Bloqueante** (para alerta temprana) | Sin anticipación; latencia ≥ 1 h incluso con dato horario | Alta | E3 y caso IN3 | No es corregible con este dato; redefinir el uso |
| B4 | Alto | El dato mide validaciones de pago (gratuidad, evasión, fallas de torniquete alteran la medida) | Alta | MED1 (elecciones) | Calendario oficial de gratuidad y estado del recaudo como contexto |
| B5 | Alto | Sensibilidad moderada (D2 incumplido): aumentos por evento × 1,5 → 46 %; estacionales no modeladas casi invisibles | Alta | F4, E3, E4 | Contexto multilínea, eventos programados, modelo intra-día calibrado |
| B6 | Alto | Falsa sensación de seguridad ante "Sin anomalía" | Alta | IN1, IN2 | Mitigado en la UI (texto explícito); capacitación |
| B7 | Medio | Alertas recurrentes sin supresión ni reconocimiento (fatiga) | Media | E4 recurrente: 100 % de las ocurrencias | Flujo de reconocimiento y supresión de causas conocidas |
| B8 | Medio | Festivos de temporada (Navidad, Año Nuevo) generan alertas (D3 = 11,1 %) | Media | F4 | Tipo de día propio, evaluado prospectivamente |
| B9 | Medio | App sin autenticación ni control de acceso; Streamlit sin endurecer para multiusuario | Media | Configuración | Desplegar detrás de SSO o proxy, TLS y registro de acceso |
| B10 | Bajo | Accesibilidad residual de componentes de Streamlit/Vega; textos del marco en inglés | Baja | axe | Componentes propios o otra versión del marco |
| B11 | Bajo | Recalibración del umbral no validada prospectivamente | Baja | E6b (exploratorio) | Evaluarla en el modo sombra |

## 4. Cambios necesarios para alcanzar un nivel operativo confiable (en orden)
1. **Datos:** acceso al flujo de validaciones por estación cada 15 min o menos, con latencia de minutos; bitácora del Centro de Control; calendario de gratuidad, horarios especiales y eventos masivos.
2. **Validación:** modo sombra prospectivo de 6 meses según el protocolo, con criterios de aceptación acordados con el Metro.
3. **Modelo:** perfiles por estación y franja de 15 min; contexto multilínea (sistémico frente a local); eventos y gratuidad como covariables; política de umbral y recalibración evaluada en el modo sombra; manejo de cambios de régimen.
4. **Operación:** orquestación diaria y horaria (`run_daily.py` programado), notificaciones, flujo de reconocimiento y clasificación humana, panel de monitoreo (tasa de alertas, deriva, calidad), autenticación y despliegue endurecido.
5. **UX:** pruebas con usuarios del Centro de Control; localización completa; vista por estación cuando haya datos.

## 5. Inventario de pruebas y resultados
| Grupo | Tipo | Pruebas | Resultado | Evidencia |
|---|---|---|---|---|
| Unitarias (datos, calendario, detectores, inyección, evaluación, etiquetas, artefactos, utils legacy) | [AUTO] | ver `pytest --collect-only` | ✔ | CI GitHub Actions + local |
| Integración (datos reales, servicio, operación diaria, AppTest, demos legacy) | [AUTO] | idem | ✔ | idem |
| Contraste de auditoría (C1, C2, C5, A1, A2) | [AUTO] | 5 | ✔ | `tests/audit/` |
| Autoencoders (causalidad, determinismo) | [AUTO] | 4 | ✔ (job `research` de CI) | `tests/unit/test_detectors_nn.py` |
| E2E app (latencia, carga, error controlado) | [AUTO] navegador | 1 | ✔ p95 669–685 ms | `tests/e2e/test_e2e.py` |
| Auditoría UX completa (35 pasos, axe, responsivo, UI = backend) | [AUTO] navegador | 1 | ✔ | `tests/e2e/test_ux_audit.py`, `reports/ux/` |
| Simulación de producción E1–E7 | [AUTO] experimento | 7 bloques | ver §2 del informe F8/F9 | `reports/produccion/validacion_produccion.json` |
| Determinismo (D9) | [AUTO] | 8 salidas | ✔ | `reports/determinismo.md` |
| Inspección visual de capturas (histórico, carga, móvil, gráficas) | [VISUAL] | 16 defectos documentados (antes/después) | 16 corregidos; 5 riesgos residuales | `docs/validacion/img/ux/` |
| Eventos reales con fuentes oficiales | [FUENTE] | 3 incidentes + 1 efecto de medición | 1/3 incidentes detectados | `reports/produccion/fuentes/` |
| Usuarios reales, lectores de pantalla, otros navegadores, carga multiusuario | [PENDIENTE] | — | no ejecutado | — |
| Validación frente a la bitácora del Centro de Control | [PENDIENTE] — **bloqueante** | — | no ejecutable sin datos | protocolo |

Los conteos exactos por archivo de la ejecución final están en la bitácora (entrada F9.5).

## 6. Explicación en lenguaje claro
El sistema toma, para cada día, la cantidad de personas que entraron a la Línea A en cada hora (de 4 a. m. a 10 p. m.), según el registro de pasajes validados. Calcula cuánto se esperaría ver en cada hora para ese tipo de día —hábil, sábado, domingo o festivo, Semana Santa o temporada de fin de año— ajustado al nivel de demanda de las cuatro semanas anteriores. Luego mide qué tan lejos está cada hora de lo esperado, comparando esa distancia con la variación que esa misma hora suele tener. Una variación normal es una diferencia del tamaño de las que ocurren casi todos los días. Un verdadero valor atípico es una diferencia tan grande, en pocas horas o en todo el día, que en el periodo de referencia solo se vio en el 5 % de los días o menos; solo entonces el sistema emite una alerta. Esa alerta es una **anomalía detectada**: un hecho estadístico que dice "la afluencia se comportó distinto de lo habitual", y no dice por qué. Puede ser una falla de captura del dato, un festivo especial, un día sin cobro de pasaje, un evento masivo o un problema en la operación. Pasa a ser una **sospecha operacional** cuando una persona revisa la alerta, descarta el calendario, la gratuidad y los errores de datos, y encuentra un patrón compatible con un problema de servicio. El caso típico es una caída brusca de las entradas en horas pico, concentrada en pocas horas, en un día hábil sin otra explicación, como el 28 de agosto de 2023 entre las 5 y las 8 p. m. Solo es un **incidente confirmado** cuando lo respalda un registro operativo del Metro (Centro de Control o comunicado oficial); el sistema por sí solo nunca lo confirma. Además, puede no alertar ante incidentes reales que no cambian cuánta gente entra a la línea, como una operación parcial en algunas estaciones, y no avisa antes de que ocurran: detecta la desviación después, cuando los datos de esas horas ya están disponibles.
