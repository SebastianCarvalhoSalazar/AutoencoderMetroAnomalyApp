# Informe de validación — F8 (preparación para producción) y F9 (frontend y UX)

Fecha: 2026-11-23 · Criterios pre-registrados en `docs/PLAN_MAESTRO.md` §9 (commit `81adf8a`, antes de ejecutar pruebas).
Leyenda de evidencia: **[AUTO]** prueba automatizada reproducible · **[VISUAL]** inspección visual de capturas por el auditor · **[FUENTE]** verificación con fuente oficial · **[PENDIENTE]** no verificable con los recursos disponibles.

---

## 1. Qué se validó y con qué datos
| Elemento | Detalle |
|---|---|
| Sistema | `perfil_contextual/peak_shift`, umbral 7,388 (congelado en DEV), artefacto `artifacts/modelo.json` |
| Datos | Línea A, conteos horarios de entradas 04–22 h. Simulación sobre el HOLDOUT 2023-07-01…2024-06-30 (364 días completos); fuentes oficiales para eventos reales |
| Política simulada | Reentrenamiento mensual con datos < 1.er día del mes; puntuación de fin de día con contexto < t; intra-día con horas ≤ h (**supuesto no verificado:** que exista un dato horario disponible al cierre de cada hora; el dato real es un export diario) |
| Scripts | `scripts/production_validation.py` → `reports/produccion/validacion_produccion.json`; `scripts/run_daily.py`; `src/metro_anomaly/simulation.py`, `operations.py` |
| Frontend | `tests/e2e/ux_audit.js` (Playwright + axe-core 4.10.2), `tests/e2e/check_ux_audit.py` (UI frente a backend), `tests/e2e/app.e2e.js`, `tests/e2e/mobile_full.js`; 17 archivos de prueba generados por `tests/e2e/make_ux_fixtures.py` |

## 2. Simulación cronológica y operación (backend)

### 2.1 Disponibilidad temporal y procesamiento (PR1, PR2) — [AUTO]
| Métrica | Resultado | Criterio |
|---|---|---|
| Violaciones de la guarda de disponibilidad (filas con fecha > t en el contexto) | **0** de 364 días | PR1 ✔ |
| Puntuación con contexto truncado = puntuación por lote = evaluación de F4 | idénticas (tolerancia 1e-12) | PR1 ✔ |
| Tiempo por día (puntuación con guarda) | 0,040 s | PR2 ✔ (< 1 s) |
| 12 reentrenamientos mensuales | 1,28 s | PR2 ✔ |
| 12 líneas, ajuste + puntuación de 6 meses | 1,86 s (viabilidad técnica; el modelo **no** está calibrado ni validado fuera de la Línea A) | PR2 ✔ |
| Línea L (no opera todas las horas) | antes: `ValueError` opaco; ahora: "Historia insuficiente: no hay días completos…" | defecto corregido (test de regresión) |

### 2.2 Estabilidad de las alertas (PR5) — [AUTO]
Al repuntuar cada día con el modelo del mes siguiente (que ya incluye ese mes), la concordancia de la decisión es **99,73 %**: solo cambia el 2024-01-12, y la correlación de scores es 0,997 → **PR5 ✔**.

### 2.3 Latencia intra-día (PR3) y anticipación (PR6) — [AUTO]
Falsas alarmas de días parciales: con el umbral de fin de día, el % acumulado de días reales que alguna vez superan el umbral pasa de 2,7 % (tras las 04 h) a **3,6 %** (tras las 22 h), frente a 2,2 % al cierre del día; el 1,4 % de los días tendría una alerta intra-día que no persiste al cierre → ≤ 5 % ✔.

| Escenario inyectado en días reales (inicio aleatorio 06–20 h) | Detección en el día | Retardo mediano (p90) desde el inicio | Comentario |
|---|---|---|---|
| Abrupta severa (× 0,2 desde la hora de inicio) | 98,9 % | **1 h** (1 h) | PR3 ✔ |
| Abrupta moderada (× 0,5) | 83,0 % | 1 h (1 h) | |
| Aislada severa (una hora × 0,2) | 98,6 % | 1 h | |
| Aislada moderada (una hora × 0,5) | 74,5 % | 1 h | |
| Progresiva intra-día (de 1,0 a 0,3) | 73,6 % | **7 h** (10 h) | el 42,9 % se detecta antes de llegar a −50 % |
| Aumento por evento × 1,5 (3 h) | 45,9 % | 1 h (2 h) | débil |
| Aumento por evento × 2 (3 h) | 77,7 % | 1 h | |

"Retardo 1 h" = la alerta es posible al cierre de la primera hora afectada (con datos horarios). **PR6 (anticipación antes del inicio): no demostrada.** Las alarmas anteriores al inicio son falsas por construcción (3,3–3,6 %, igual a la tasa de falsas alarmas). El sistema no contiene ningún mecanismo de predicción.

### 2.4 Batería de escenarios multi-día — [AUTO] (6 fechas de inicio, reentrenamiento mensual sobre los datos modificados)
| Escenario | Detectado (de 6) | Primer día con alerta (mediana) | % de días afectados con alerta | Lectura |
|---|---|---|---|---|
| Progresivo: −2,5 %/día durante 14 días y luego estable en −35 % | 6/6 | día 7 (≈ −17,5 %) | 24,6 % | se detecta cuando la caída acumulada ya es grande |
| Escalón sostenido −20 % (60 días) | 5/6 | día 0 | 8,3 % | **el nivel causal absorbe el cambio en pocos días**: las alertas se apagan aunque el problema persista |
| Escalón sostenido +20 % | 4/6 | día 10 | 6,1 % | aumentos sostenidos, poco visibles |
| Recurrente: lunes 17–18 h × 0,5 durante 10 semanas | 6/6 | día 0 | **100 %** | nunca lo "aprende": alerta en cada ocurrencia (no hay supresión ni reconocimiento) |
| Estacional no modelada: −15 % durante 21 días | 2/6 | día 6,5 | 2,4 % | casi invisible |
| Contextual: perfil festivo en día hábil | — | — | 100 % (242 días) | ✔ |
| Contextual: perfil hábil en domingo/festivo | — | — | 100 % (69 días) | ✔ |
| Contextual: perfil de sábado en día hábil | — | — | 93,8 % (242 días) | ✔ |

### 2.5 Sensibilidad, precisión, FP y FN en flujo mixto — [AUTO] (5 semillas; inyección única por día: T1/T2/T3 moderadas, T1 severa, T4)
| Prevalencia | Inyectados | TP | FN | FP | Sensibilidad | Precisión | FP/mes |
|---|---|---|---|---|---|---|---|
| 2 % | 35 | 23 | 12 | 43 | 0,657 | **0,348** | 0,72 |
| 5 % | 90 | 60 | 30 | 41 | 0,667 | 0,594 | 0,68 |

Casi todos los FP son las 8 alertas reales del año, repetidas en cada semilla. No tienen etiqueta: 1 es un incidente verificado (28/08/2023), 1 es un efecto de medición electoral, 2 son festivos y las demás no tienen explicación verificada. Si esas 8 fueran eventos reales, la precisión sería 0,885 / 0,968. **La precisión real es desconocida** porque no existe un registro de incidentes.

### 2.6 Robustez (PR4) — [AUTO]
| Prueba | Resultado |
|---|---|
| 1, 3 o 6 horas faltantes aleatorias en todos los días | tasa de alertas 2,2 % / 2,2 % / 1,65 % (sin faltantes: 2,2 %); scores siempre finitos |
| Dato retrasado (faltan los 1, 3 o 7 días previos al puntuar) | diferencia media del score 0,04 / 0,09 / 0,15; **0 cambios de decisión** en 324 días |
| Día corrupto (unidad × 10, día en ceros, × 1e4) | siempre alerta ese día; **0 alertas extra** en los 28 días siguientes (sin contaminación) |
| Reglas de calidad de datos | "cero sospechoso" y "valor implausible" (> 1,5 × máximo histórico) → categoría "Posible problema de datos" |
| Envejecimiento (modelo congelado a 2023-06 frente a reentrenamiento mensual) | alertas 2,75 % frente a 2,2 %; recall T1 moderada 0,80 frente a 0,75 → sin degradación apreciable en 12 meses |
| Recalibración del umbral (EXPLORATORIA: política definida con el holdout ya visto) | umbrales de 5,25 a 7,58; duplica las alertas en diciembre (20,7 % frente a 10,3 %). No se adopta |
| Artefacto ausente o corrupto, archivo inválido (CLI) | códigos de salida 3 y 2 con mensaje; tests `test_operations.py` |

Hallazgo de datos (nuevo): en 2019 hay días en los que las horas faltantes **se acumularon en la hora siguiente** (p. ej., 16/02/2019: sin dato de 11 a 14 h y 254.265 entradas a las 15 h). Las "anomalías" más extremas de 2019 (25/04, 04/09) son de este tipo: **artefactos de captura, no eventos**. Ahora se marcan como posible problema de datos.

### 2.7 Monitoreo, trazabilidad, escalabilidad y recuperación (PR8) — [AUTO]
- `run_daily.py` registra cada decisión en JSONL: categoría, score, umbral, severidad, horas, banderas, SHA-256 del artefacto y de los datos, versión y hora UTC. Es idempotente (reprocesar no duplica).
- `monitor()` calcula la tasa de alertas por mes frente al rango 2–10 %, los problemas de datos, las versiones del artefacto y las líneas dañadas del log.
- Corrección: el aviso "90 días sin alertas" se disparaba con probabilidad ≈ 13 % aun funcionando bien; pasó a 180 días (≈ 2 %).
- Recuperación: el artefacto es JSON versionado con hash; el pipeline completo es determinista (`reports/determinismo.md`) y se reconstruye con `make reproduce`.
- **No implementado:** orquestación programada, canal de notificación, autenticación de la app, alta disponibilidad.

## 3. Eventos reales: ¿las anomalías corresponden a incidentes? — [FUENTE]
Búsqueda en plataformas oficiales colombianas (metrodemedellin.gov.co, medellin.gov.co y telemedellin.tv, el canal público de Medellín). El acceso directo requirió un entorno con esos tres hosts, aprobado por el usuario. Control de *prompt injection*: sin instrucciones incrustadas (detalle en `reports/produccion/fuentes/extractos_fuentes_oficiales.md`).

| Id | Hecho verificado | Señal en los datos de la Línea A | ¿Alerta? |
|---|---|---|---|
| IN1 | 05/02/2023 (dom), desde ~18:41 la Línea A solo opera entre Niquía y Poblado | max \|z\| 1,3; score 0,49 | **No** |
| IN2 | 22/08/2023 (mar), desde las 07:05 opera en dos tramos; 4 estaciones centrales cerradas | 07 h z −2,6, 08 h z +2,6; score 0,61 | **No** |
| IN3 | 28/08/2023 (lun), 17:30 atrasos; luego suspensión de A, B, J, K y P; 19:43 sin restablecer | 17 h −12,6; 18 h −45; 19 h −66; score 65 | **Sí** (con datos horarios: al cierre de las 17 h, ~30 min **después** del aviso oficial de las 17:30) |
| MED1 | Jornadas electorales: **no cobro** del pasaje (desde 2006; el 29/10/2023 hasta las 17:00) | entradas ≈ 0 de 04 a 16 h en todas las líneas | sí, pero es un **efecto de medición**: el dato cuenta validaciones de pago, no pasajeros |

**Conclusiones:**
1. Sensibilidad observada frente a incidentes reales verificados: **1 de 3**. Las interrupciones parciales no dejan una huella detectable en las entradas agregadas de la línea: los usuarios entran por otras estaciones o redistribuyen su viaje.
2. Las "detecciones" electorales de D7 (EV3, EV4) no son caídas de demanda: se reclasificaron como "Efecto de medición conocido".
3. **No es posible demostrar que las anomalías detectadas correspondan a incidentes verdaderos** sin un registro independiente de incidentes. Las 8 alertas del holdout incluyen: 1 incidente verificado, 1 efecto de medición, 2 festivos de calendario y 4 sin explicación verificable (23/09, 09/12, 14/12/2023 y 11/01/2024). Protocolo propuesto: `docs/PROTOCOLO_VALIDACION_OPERACIONAL.md`.
4. Sesgo reconocido: IN3 se buscó a partir de una alerta. IN1 e IN2 salieron de búsquedas sistemáticas no guiadas por el modelo.

## 4. Frontend y UX

### 4.1 Recorrido funcional — [AUTO] (`ux_audit.js`, 35 pasos, Chromium)
- **Histórico:** carga inicial; 11 fechas (normal, incidente, Navidad, elecciones, horas faltantes 2023/2019, día sin datos, fecha mínima, fecha máxima, Semana Santa, aumento); fecha escrita fuera de rango; expansor de detalle; navegación con teclado.
- **Carga:** pestaña vacía y 17 archivos (normal xlsx/csv, anomalía inyectada, horas faltantes, fechas futuras con contexto, historia insuficiente, otra línea, negativos, esquema inválido, binario corrupto, vacío, extremos × 1000, día en ceros, texto en números, filas duplicadas, archivo oficial 2024 con 12 líneas, extensión no soportada); selector de día; descarga CSV.
- **Responsivo:** 390 × 844 y 390 × 4200.
- **Cada fecha se verifica después de escribirla:** el script confirma que el valor del campo es el esperado.

| Criterio | Resultado final |
|---|---|
| UX1 errores de JS / `pageerror` / excepciones / peticiones fallidas | **0 / 0 / 0 / 0** en los 35 pasos ✔ |
| UX2 flujos con el resultado esperado | todos ✔ (detalle en `reports/ux/auditoria_ux_despues.json`) |
| UX3 UI = backend | **100 %**: 4 métricas, categoría, horas y tabla de detalle en las 11 vistas; CSV descargado idéntico byte a byte en 3 cargas (61, 90 y 182 días) ✔ (`reports/ux/consistencia_ui_backend.json`) |
| UX4 accesibilidad (axe, WCAG 2.1 A/AA) | antes: contraste (5–10 nodos) y nombres de enlace atribuibles a la app; **después: 0 violaciones atribuibles a la app** ✔. Persisten violaciones de componentes internos de Streamlit y Vega (`aria-allowed-attr`, `button-name`, `summary-name`, `nested-interactive`) — riesgo residual |
| UX5 responsivo | sin desbordamiento horizontal (scrollWidth = clientWidth = 390) ✔; leyenda corregida para 390 px [VISUAL] |
| UX6 comunicación de la alerta | categorías "Sin anomalía estadística", "Anomalía estadística", "Posible problema de datos" y "Efecto de medición conocido"; severidad empírica; frecuencia histórica; texto "No es un incidente confirmado: requiere verificación…" verificado por AppTest ✔ |
| UX7 gráficas | leyenda, unidades (pasajeros), formato colombiano, horas sin dato marcadas, banda recortada con aviso, umbral y evolución temporal [VISUAL] ✔ |
| Latencia (A5, medida de nuevo) | p95 = 669–685 ms con cambio de fecha **verificado** ✔ |

### 4.2 Defectos encontrados, causa, corrección y verificación
Capturas: `docs/validacion/img/ux/antes/` (estado inicial) y `docs/validacion/img/ux/despues/` (tras las correcciones).

| # | Severidad | Pantalla / escenario | Esperado | Observado (antes) | Causa | Corrección | Verificación |
|---|---|---|---|---|---|---|---|
| UX-01 | Alta | Histórico, cualquier alerta (`antes/H03`) | Distinguir una anomalía estadística de un incidente; severidad y confianza | "⚠️ ANOMALÍA" en rojo, sin severidad; "✅ Normal" sin matices | Redacción binaria | Categorías, severidad empírica, frecuencia histórica, aviso explícito y siguiente paso; color ámbar | `despues/H03`, AppTest |
| UX-02 | Alta | Elecciones 29/10/2023 (`antes/H05`) | Explicar la caída | "ANOMALÍA" −99 % | El dato no registra entradas sin cobro | Categoría "Efecto de medición conocido" con fuente oficial | `despues/H05`, test de servicio |
| UX-03 | Media | Carga: día en ceros / × 1000 (`antes/U_13`, `U_12`) | Sospechar un error de captura | Solo "ANOMALÍA" | Sin reglas de calidad de datos | Reglas "cero sospechoso" y "valor implausible" | `despues/U_13`, `U_12`, tests |
| UX-04 | Media | Gráfica horaria (`antes/H06`) | Leyenda, unidades, faltantes visibles, escala legible | Sin leyenda; banda hasta 190.000 que aplastaba la serie; la hora sin dato no se distinguía | Diseño inicial | Leyenda, unidades, formato es-CO, marcador "Sin dato", recorte con aviso | `despues/H06`, `H14` [VISUAL] |
| UX-05 | Media | Ambas pestañas | Ver la evolución temporal | Sin contexto temporal | Faltaba la vista | Línea de tiempo del score (60 días o periodo cargado) con umbral | `despues/H14` [VISUAL] |
| UX-06 | Media | Detalle por hora | Tabla accesible y verificable | Tabla en canvas (no legible por lectores de pantalla) | `st.dataframe` | `st.table` | UX3 ✔ |
| UX-07 | Media | Carga | Exportar resultados | No existía | — | Botón "Descargar resultados (CSV)" | CSV idéntico al backend |
| UX-08 | Media | Global (axe) | Contraste ≥ 4,5:1 | `code` y pie gris en la barra lateral; rojo primario; encabezados de tabla #83858c (3,68:1) | Tema por defecto | Tema `primaryColor`/`textColor`, sin `code` ni pie de página | axe: 0 violaciones atribuibles |
| UX-09 | Baja | Campo de fecha | dd/mm/aaaa | aaaa/mm/dd y fechas ISO mezcladas | Valores por defecto | `format="DD/MM/YYYY"` y fechas es-CO | capturas |
| UX-10 | Baja | Encabezado del día | Etiquetas legibles | `habil`, `perfil_contextual` en formato código | Códigos internos | Etiquetas en español | capturas |
| UX-11 | Baja | Cargador | Límite coherente | "Limit 200MB" frente a 20 MB del servicio | Configuración | `maxUploadSize = 20` | `despues/U_13` |
| UX-12 | Baja | Barra superior | Sin opciones de desarrollo | Botón "Deploy" | Configuración | `toolbarMode = "viewer"` | capturas |
| UX-13 | Baja | Fecha escrita fuera de rango (2018) | Mensaje de rango | "No hay datos para…" | `min_value` no se aplica al texto escrito | Validación explícita del rango | `despues/H13`, AppTest |
| UX-14 | Baja | Archivo .txt tras uno válido (`antes/U_17`) | Saber a qué archivo corresponden los resultados | Resultados del archivo anterior sin indicarlo | Comportamiento de Streamlit | Encabezado "Resultados de «archivo»" | `despues/U_17` |
| UX-15 | Baja | Encabezados (axe `link-name`) | Enlaces con nombre | Anclas vacías | Anclas automáticas | `anchor=False` | axe |
| UX-16 | Baja | Móvil, leyenda | Leyenda completa | "Rango norm…" cortado | Leyenda horizontal | 2 columnas y nombres cortos | `despues/M03` [VISUAL] |
| T-01 | **Prueba** | E2E de F5 | Cambiar la fecha | `fill()` no aplicaba la fecha: la latencia de F5 medía reejecuciones sin cambio (**evidencia inválida**) | Campo con máscara | Escritura carácter a carácter + verificación | p95 685 ms con cambio verificado |
| T-02 | **Prueba** | Auditoría | Fecha aplicada | La máscara conservaba un dígito ("37/03/2024"): H11–H13 mostraban resultados obsoletos | Máscara | Vaciado completo + reintentos + bandera `applied` | 12/12 fechas aplicadas |

### 4.3 Riesgos residuales de UX (no corregidos)
- Violaciones de accesibilidad de componentes internos de Streamlit/Vega (menú, botón de colapsar, menú de acciones de las gráficas, zona de carga).
- Textos del marco en inglés ("Drag and drop file here", "Browse files"); Streamlit 1.39 no permite traducirlos.
- La tabla de resultados usa punto decimal (columnas numéricas ordenables); el resto de la app usa coma.
- Los símbolos de la leyenda no distinguen línea continua, discontinua y área.
- La banda horaria es ancha en las horas pico (una hora aislada debe desviarse ~±50 %): es veraz, pero puede leerse como insensibilidad.
- [PENDIENTE] Pruebas con usuarios reales del Metro (comprensión de las alertas, flujo de trabajo); lectores de pantalla reales; navegadores distintos de Chromium.
