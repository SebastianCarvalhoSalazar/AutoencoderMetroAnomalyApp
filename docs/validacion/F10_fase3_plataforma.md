# Validación de la fase 3 — plataforma de analítica operacional multilínea

Fecha: 2026-11-24 · Artefactos: `artifacts/plataforma/` (código `c1e0481`, SHA-256 de los datos en `manifiesto.json`).

## 1. Qué se construyó
| Componente | Archivo | Tipo de salida |
|---|---|---|
| Inventario y EDA multilínea | `scripts/eda_multilinea.py` → `reports/fase3/eda_multilinea.md` | Descriptivo |
| Topología verificada (líneas y estaciones de integración) | `data/topologia_red.csv`, `data/lineas.csv` | Dato de referencia (URL oficial por fila) |
| Fuente externa de clima (alineada por hora) | `src/metro_anomaly/fuentes/clima.py` | Investigación (H5b descartada) |
| Índice de red y detector de coherencia | `src/metro_anomaly/red.py` | Estimación |
| Clasificador de alcance y regla de ausencia simultánea | `src/metro_anomaly/red.py` | Estimación (alcance: orientativo) |
| Pronóstico del día siguiente con intervalo | `src/metro_anomaly/pronostico.py` | Predicción |
| Orquestador, motor de reglas y manifiesto de procedencia | `src/metro_anomaly/plataforma.py`, `scripts/fase3/build_plataforma.py` | Real + estimación + predicción |
| Página interactiva | `pages/2_Red_multilinea.py` | Distingue real, estimación, simulación y predicción |

## 2. Resultados frente a criterios pre-registrados
Detalle en `docs/PLAN_MAESTRO.md` §10 y en `reports/fase3/`.

| Hipótesis | Veredicto | Cifra clave |
|---|---|---|
| H1 — coherencia entre líneas | **Cumple** (adoptado en A, B y 1) | Línea A: recall moderado 0,702 → 0,900; diferencia +0,198 [0,177; 0,216] con el mismo presupuesto de alertas |
| H2 — local vs. generalizado | **No cumple** (solo orientativo) | Exactitud 0,884 < 0,90 |
| H3 — redistribución | Descriptivo, n = 1 | 28/08/2023, 19 h: la Línea 2 sube con z = 6,8 |
| H4 — ausencia simultánea en la red | **Cumple** (reinterpretada) | Recall 1,0; 4/731 días reales, todos con causa operacional o de calendario |
| H5 — pronóstico del día siguiente | Puntual: **cumple**. Intervalo sin calibrar: **no cumple** | WAPE 11,5–27,6 % menor que el naive en 11/11 líneas. Cobertura sin calibrar en A: 0,723. La banda que muestra la plataforma (calibración conformal, centrada en el modelo elegido) cubre 0,751–0,831 |
| H5b — lluvia | **Descartada** | La lluvia observada empeora el WAPE en las 11 líneas |
| H6 — días de alta demanda | **Parcial** (7/11 líneas) | AUC de 0,84 a 0,97 en A, 2, J, M, O, P y T-A |
| H7 — demografía y suelos | **Descartada** | No alineables con línea × hora |

**Evidencia más débil (declarada):**
- La calibración del intervalo y los candidatos `perfil` y `combo` del pronóstico se añadieron después de ver el holdout v1.
- En el caso del 28/08/2023 de H2, el criterio se fijó habiendo visto los z del EDA.
- El periodo 2023-07…2024-06 ya se había inspeccionado para la Línea A en las fases 1–2.

## 3. Pruebas
| Nivel | Archivo | Qué verifica | Resultado |
|---|---|---|---|
| Unitarias | `tests/unit/test_red.py` (6) | El detector de coherencia descuenta choques compartidos, mantiene los locales y es causal; clasificador de alcance; monitor de ausencias | 6/6 |
| Unitarias | `tests/unit/test_pronostico.py` (4) | Rasgos causales, intervalos ordenados, WAPE, alineación horaria de la lluvia | 4/4 |
| Unitarias | `tests/unit/test_plataforma.py` (6) | Prioridad de reglas (elecciones solo en modos sin cobro, ausencia simultánea frente a falla de una línea, local frente a generalizada) y causalidad de la reproducción | 6/6 |
| Integración | `tests/integration/test_red_page.py` (4) | La página coincide con los artefactos (11 líneas), elecciones, pestañas y catálogo, simulación | 4/4 |
| E2E con navegador | `tests/e2e/red_multilinea.js` + `check_red_multilinea.py` (en `test_e2e.py`) | 4 fechas reales, reproducción por hora (08, 17 y 19 h), 4 pestañas, móvil (iPhone 13), axe, tiempos | 44/44 celdas idénticas a los datos; 0 excepciones; 0 desbordamientos; 0 errores de JavaScript |
| Suite rápida completa | `make test` | Regresión de las fases 1–2 | 174 aprobadas, 1 omitida (TensorFlow), cobertura del 83 % |
| E2E completo | `pytest -m e2e tests/e2e/test_e2e.py` | App original y página nueva | 2/2 |

**Rendimiento** (AppTest, servidor):
- Primera carga: 1,85 s.
- Cambio de fecha: mediana 0,36 s y p95 0,48 s (20 fechas).
- En el navegador el cambio de fecha mide unos 4 s, pero incluye unos 3 s de esperas fijas del script E2E.
- La reproducción por hora ocurre en el navegador (control deslizante de Vega), sin recargar la página.
- Construcción de los artefactos: unos 12 minutos (24 meses × 11 líneas, ajustes causales mensuales).

**Accesibilidad (axe, WCAG 2 A/AA):** sin violaciones atribuibles a la página después de las correcciones. Las restantes pertenecen al marco: el menú de Streamlit (`#MainMenu`) y el menú de acciones de Vega (`summary-name`).

## 4. Defectos hallados y corregidos durante la fase
| # | Defecto | Cómo se detectó | Corrección |
|---|---|---|---|
| D1 | Comparación de modelos de pronóstico sobre filas distintas (el naive no tiene referencia en días incompletos de los cables) | Revisión de NaN por línea | Métricas solo sobre filas comunes |
| D2 | Un cero en otra línea (falla de datos) arrastraba el índice de red y generaba una anomalía local falsa | `test_plataforma.py` | Los ceros se excluyen del índice y del alcance; H1–H4 se re-ejecutaron (resultados idénticos) |
| D3 | El pronóstico del día futuro fallaba porque `profile_predict` descartaba fechas sin dato | Construcción de artefactos | Se admiten fechas posteriores al último dato |
| D4 | Contraste insuficiente (2,1:1 y 2,9:1) en textos secundarios | axe | Texto normal sin código en línea |
| D5 | La Línea L (sin datos) desaparecía del esquema | Revisión visual | Capa base gris para todos los nodos |
| D6 | Un gráfico de 1 051 px quedaba recortado en escritorio y en móvil | Nueva verificación E2E de recorte dentro de gráficos y tablas | Un gráfico por línea, del ancho del contenedor |
| D7 | La tabla de 6 columnas no cabía en 390 px | Captura móvil | 4 columnas, con el detalle combinado |
| D8 | Una prueba tardó 606 s por sobresuscripción de hilos (OpenMP) mientras corría otra construcción | Duraciones de pytest | No es un defecto del código: 3 s en solitario. No ejecutar construcciones y pruebas en paralelo |

## 5. Catálogo de oportunidades (estado al cierre)
| Oportunidad | Estado | Requisito para avanzar |
|---|---|---|
| Revisión retrospectiva por línea con descuento de efectos de red | **Operable** (A, B y 1 con coherencia; resto con perfil) | — |
| Ausencia simultánea de registro (operación, calendario o datos) | **Operable** | Bitácora de incidentes para atribuir la causa |
| Pronóstico del día siguiente por línea y hora | **Operable** con intervalo calibrado | Seguimiento mensual de la cobertura |
| Indicador de día de alta demanda | **Operable en 7 líneas** | Más años de historia para B, K y H |
| Alcance local / multilínea / generalizado | Orientativo | Etiquetas reales de incidentes |
| Redistribución entre líneas | Descriptiva | Datos por estación y matriz Origen-Destino |
| Detección temprana, localización, mantenimiento predictivo, capacidad y frecuencia | **No viable con los datos actuales** | Ver `docs/fase3/HOJA_DE_RUTA_DATOS.md` |

## 6. Dictamen de la fase 3
La plataforma es **apta con condiciones** para:
- revisión retrospectiva multilínea;
- control de ausencias de registro;
- pronóstico del día siguiente por línea y hora, con incertidumbre.

Sigue **no apta** para detección temprana de incidentes, localización de fallas o mantenimiento predictivo. La causa son los datos (línea × hora, export diario), no los modelos.

**Condiciones:**
- revisar mensualmente la cobertura del intervalo y la tasa de alertas por línea;
- reconstruir los artefactos con `scripts/fase3/build_plataforma.py` cada vez que lleguen datos nuevos y verificar `git_commit_build` en el manifiesto;
- no presentar el alcance ni la redistribución como hallazgos validados.

## 7. Cierre de pendientes y evaluación de extremo a extremo de la operación (F10.8)
| Pendiente | Implementación | Evaluación |
|---|---|---|
| Operación diaria multilínea | `src/metro_anomaly/operacion_red.py`; `run_daily.py --red [--pronostico] [--monitor]`; auditoría JSON Lines idempotente; monitoreo por línea (tasa de alertas, cobertura del intervalo, días sin registro, cierres) | 14 pruebas de integración. Simulación cronológica de junio de 2024: **319/319** línea-días iguales a la reproducción histórica y sin advertencias del monitor |
| Pronóstico operativo | `pronostico.forecast_day` con reajuste diario, banda calibrada con el factor conformal del manifiesto y calendario de cierres | Reajuste diario frente a mensual: WAPE equivalente en las 11 líneas (A: 0,0347 frente a 0,0346); cobertura del 80 % entre 0,797 y 0,901 (`reports/fase3/operacion_red.md`) |
| Determinismo de la plataforma | `scripts/fase3/check_determinism_plataforma.py` | Dos construcciones aisladas, idénticas entre sí y a las versionadas (`reports/fase3/determinismo_plataforma.md`) |
| Reproducibilidad desde un clon limpio | Las predicciones fuera de muestra se versionan en `reports/fase3/pronostico_oos_holdout.csv.gz` (antes dependían de un `.pkl` ignorado por git) | La construcción de la verificación de determinismo parte de copias sin `.git` ni `.pkl` |
| Cierre del 24 y 31 de diciembre | `data/eventos_reales.csv` OP1–OP3; nota en `event_note` | Oficial en 2022 y 2024 (22:00); **2023 solo prensa** (21:00), coherente con los datos |
| Animación de la reproducción | Botón "▶ Reproducir el día hora a hora" | E2E: 5/5 fotogramas distintos |
| Fichas del modelo y de los datos | `docs/MODEL_CARD.md`, `docs/DATA_CARD.md` | — |

**Defectos hallados por la evaluación de extremo a extremo y corregidos:**

| # | Defecto | Cómo se detectó | Corrección y prueba |
|---|---|---|---|
| D9 | Una evaluación que empezaba a mitad de mes ajustaba con datos hasta t−1 en lugar de hasta el inicio del mes. La Línea 1 del 08/06/2024 perdía la alerta (score 5,80 → 4,66) | Simulación: 309/310 coincidencias | Ajuste siempre con datos anteriores al mes. `test_mid_month_start_matches_full_month` falla sin la corrección |
| D10 | **Falla silenciosa:** una línea sin ningún registro en el día no generaba fila (M del 14 al 16 y H del 17 al 19/06/2024, cierres de 2022) | Análisis del error de pronóstico de la Línea P (WAPE 0,28) | Categoría "línea sin registro" (la línea operó ≥ 80 % de los 56 días previos) y calendario de cierres programados verificados. 52 días sin registro y 35 días de cierre programado en 2022-04…2024-06 |
| D11 | El lote fallaba al registrar una línea sin fila en el día (`KeyError`) | Simulación tras D10 (27/06/2024, Línea K) | `reindex`; `test_line_without_records_is_logged` |

**Hallazgo operativo verificado:** la Línea P registró 300–680 entradas al día del 8 al 13/06/2024, frente a unas 22 000 habituales. Fue el mantenimiento anual anunciado por Telemedellín el 06/06/2024.
- Con el calendario de cierres anunciados, el WAPE de junio de la Línea P baja de 0,276 a 0,092.
- Los días de cierre se pronostican en 0 por regla y quedan fuera de la cobertura y del WAPE del modelo.
- Es información anunciada de antemano, sin fuga, pero depende de que alguien cargue los anuncios (fuente n.° 3 de la hoja de ruta).
- Para 2023 no se halló un anuncio con año explícito: esos cierres aparecen como "línea sin registro".

**Pruebas al cierre:**
- 194 rápidas aprobadas y 1 omitida (TensorFlow), con 84,8 % de cobertura;
- 2 E2E aprobadas;
- verificación visual-datos: 44/44 celdas;
- animación: 5/5 fotogramas distintos.

**Latencia de la operación diaria:** evaluación de 11 líneas con p95 de 9,4 s; pronóstico del día siguiente con p95 de 55 s, por el reajuste diario de los GBM con 2,5 años de datos.

**Pendiente que no depende del código:** validar la calibración del intervalo y H2/H3 con datos posteriores a junio de 2024 y con la bitácora oficial de incidentes.
