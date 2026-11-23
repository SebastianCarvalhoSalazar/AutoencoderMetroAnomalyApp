# Plan maestro de implementación — Detección de anomalías en la afluencia de la Línea A

> Documento vivo. Se actualiza **antes** de empezar cada fase (revisión de dependencias) y **después** de cada cambio (estado, evidencia). La historia cronológica está en [`BITACORA.md`](BITACORA.md); los informes de validación de cada fase, en [`validacion/`](validacion/).
> Origen: auditoría [`audit/AUDITORIA.md`](../audit/AUDITORIA.md) (commit `3a1948c`).

Estados: `PENDIENTE` · `EN PROGRESO` · `BLOQUEADA` · `COMPLETADA`. Ninguna tarea se marca `COMPLETADA` sin una evidencia verificable (test, comando reproducible, archivo o salida registrada).

---

## 1. Problema que debe resolverse (redefinido)

**Objetivo operativo:** dado el conteo horario de entradas a la Línea A (04:00–23:00), detectar **días y horas cuya afluencia se desvía de forma significativa de la esperada para ese contexto**: día de la semana, festivo, temporada y nivel reciente de demanda. La detección debe:

1. **No confundir efectos de calendario previsibles con anomalías.** Un festivo con afluencia de festivo es normal; un día hábil con perfil de festivo no lo es.
2. **Ser causal:** solo puede usar información disponible hasta el momento evaluado, para ser compatible con el monitoreo en tiempo real que plantea el README.
3. **Explicar** qué horas se desvían, en qué dirección y con qué magnitud.
4. **Adaptarse a cambios de nivel** de la demanda sin re-etiquetado manual (auditoría A3).

**Tipología de anomalías objetivo:**
- (T1) Interrupción o reducción del servicio durante algunas horas.
- (T2) Reducción o aumento de todo el día.
- (T3) Aumento por evento.
- (T4) Errores de datos (hora duplicada, hora en cero, desfase horario).
- (T5) Perfil incoherente con el calendario.

**Fuera de alcance:** atribuir causas externas (clima, eventos) sin datos que las respalden.

## 2. Principios de validación (se fijan antes de implementar)

| Principio | Mecanismo concreto | Verificación automatizada |
|---|---|---|
| Sin fuga de datos | Todo ajuste (escalas, perfiles, modelos, umbrales) usa solo datos con fecha < la fecha evaluada | Test de **causalidad**: el score del día *t* es idéntico (igualdad exacta) aunque se agreguen datos posteriores a *t* |
| Separación temporal | DEV = 2022-01-01…2023-06-30 (selección de modelo con *rolling origin* mensual). HOLDOUT = 2023-07-01…2024-06-30 (se usa **una sola vez**, con el modelo y el umbral congelados). EXTERNO = 2019 (otro régimen) y 2020–2021 (estrés COVID) | Test que verifica que la selección no lee fechas ≥ 2023-07-01; registro de los accesos al holdout en la bitácora |
| Etiquetas | Las etiquetas heurísticas **no son verdad de terreno**. Fuentes de verdad: (a) anomalías sintéticas inyectadas sobre días reales, con la verdad conocida por construcción; (b) eventos reales documentados en fuentes oficiales; (c) la heurística corregida solo como métrica de consistencia secundaria | Tests de la biblioteca de inyección; tabla de eventos con su fuente |
| Umbral | Se elige en DEV para un **presupuesto de alertas** (objetivo 5 % de los días) y se congela antes de abrir el holdout | El artefacto guarda el umbral y su procedencia (periodo y procedimiento) |
| Líneas base | Toda propuesta se compara con: naive estacional (mismo día de la semana anterior), perfil robusto contextual, AE original reentrenado sin fuga, AE condicionado, IsolationForest | Tabla comparativa con IC por bootstrap y prueba pareada |
| Reproducibilidad | Semillas fijas, dependencias fijadas e instalables en Linux, hash de los datos en los artefactos, un solo comando para reproducir todo | Doble ejecución con métricas idénticas; CI |
| Regresiones | `pytest` (unitarias, integración, end-to-end) en cada cambio; CI en GitHub Actions | Bitácora con el resultado de cada ejecución |
| Honestidad | Los resultados negativos se publican; las desviaciones del plan se registran con su justificación | Sección "Desviaciones" de la bitácora |

## 3. Definición de "terminado" (criterios pre-registrados)

Estos umbrales se fijan **antes** de ver resultados del holdout. Si no se cumplen, el resultado se informa tal cual y se reevalúa; los criterios no se ajustan a posteriori.

### 3.1 Modelo (evaluado en HOLDOUT con umbral congelado en DEV)
| ID | Criterio | Meta |
|---|---|---|
| D1 | Tasa de alertas diaria sobre días reales del holdout | entre 2 % y 10 % (objetivo 5 %) |
| D2 | Recall sobre anomalías inyectadas **severas** / **moderadas** (catálogo §5) | ≥ 0,95 / ≥ 0,80 (las **leves** se informan sin meta) |
| D3 | Tasa de alertas en festivos reales (efecto de calendario esperado) | ≤ 10 % |
| D4 | Recall del escenario T5 (perfil de festivo en un día hábil) | ≥ 0,95 |
| D5 | El modelo elegido no es peor que la mejor línea base en recall moderado con el mismo presupuesto de alertas (IC 95 % de la diferencia pareada) y supera al AE original | diferencia con el AE original > 0 con IC 95 % |
| D6 | Generalización externa: tasa de alertas en 2019 después de un calentamiento causal de 8 semanas | ≤ 15 % |
| D7 | Eventos reales verificados (fuente oficial): detección en la fecha o en los 2 días siguientes al inicio | ≥ 2/3 de los eventos verificados; si hay < 3 eventos verificables, queda como limitación explícita |
| D8 | Causalidad: score idéntico al agregar datos futuros | test en verde |
| D9 | Determinismo: dos ejecuciones completas producen métricas idénticas | test o comando en verde |

### 3.2 Pipeline de datos
- P1. Carga y valida el esquema de los 6 archivos (todas las líneas y años), con errores explícitos ante esquemas inesperados.
- P2. Duplicados entre archivos resueltos de forma determinista; reporte de calidad generado (faltantes, ceros estructurales, consistencia `Total general` = Σ horas).
- P3. La semántica de los NaN de 23:00 (y de 22:00) está decidida y documentada con evidencia.
- P4. Tests unitarios del módulo de datos con cobertura ≥ 85 %.

### 3.3 Aplicación
- A1. Sin `pickle` de usuario. Entradas: (a) selección de una fecha del histórico; (b) carga de `.xlsx`/`.csv` en el formato oficial (uno o varios días).
- A2. Valida las entradas con mensajes claros: columnas faltantes, NaN, negativos, línea inexistente, fecha sin historia suficiente. Nunca muestra una traza al usuario.
- A3. Muestra el observado frente a la banda esperada, las horas anómalas, el contexto de calendario y el score; el umbral se lee del artefacto.
- A4. Tests: `streamlit.testing.AppTest` (integración) y Playwright end-to-end (subida de archivo y selección de fecha), en verde.
- A5. Latencia p95 < 2 s al puntuar un día con la caché caliente.

### 3.4 Reproducibilidad e infraestructura
- R1. `pip install -r requirements.txt` funciona en Linux (Python 3.10) desde un entorno limpio.
- R2. `make reproduce` (o un script equivalente) regenera datos procesados, selección, evaluación, artefactos e informes.
- R3. Los artefactos incluyen metadatos: hash de los datos, commit, configuración, umbral y métricas.
- R4. CI ejecuta la suite de tests en cada push (estado verificado en GitHub Actions, si está habilitado en el repositorio).
- R5. Cobertura total de `src/` ≥ 80 %.

### 3.5 Documentación
- README alineado con el sistema real.
- Ficha del modelo (*model card*) y ficha de los datos (*data card*).
- Un informe de validación por fase y un informe final.
- El notebook y la app antiguos quedan marcados como *legacy*, con referencia a la auditoría.

## 4. Arquitectura objetivo

```
src/metro_anomaly/
  data.py         # carga y validación de los Excel oficiales → tabla larga/ancha; reporte de calidad
  calendar.py     # festivos de Colombia, días especiales, tipo de día, rasgos de calendario
  detectors/      # interfaz común fit(history) / expected(dates) / score(day)
    naive.py      # naive estacional
    profile.py    # perfil robusto contextual (mediana/MAD en log1p, ventana causal)
    regression.py # regresión de calendario + nivel reciente (candidato)
    autoencoder.py# AE original y AE condicionado (solo investigación; requiere TF)
    iforest.py
  scoring.py      # residuos estandarizados por hora, agregación al día, explicación
  injection.py    # catálogo de anomalías sintéticas con su verdad de terreno
  labels.py       # heurística IQR corregida (solo métrica secundaria) + análisis de calidad
  evaluation.py   # splits, rolling origin, métricas, bootstrap, comparación pareada
  thresholds.py   # umbral por presupuesto de alertas
  artifacts.py    # guardado y carga de modelo + metadatos
scripts/          # prepare_data.py, select_model.py, evaluate_holdout.py, build_artifact.py
app.py            # nueva app (servicio puro en src/; la UI solo orquesta)
tests/            # unit/, integration/, e2e/
configs/          # config.yaml (splits, catálogo, presupuesto, semillas)
artifacts/        # modelo final + metadatos (versionado)
reports/          # tablas y figuras generadas
legacy/           # app y utils originales (referencia histórica)
```
TensorFlow pasa a `requirements-research.txt`. Si el modelo elegido no es un AE, el runtime de la app no depende de TF (se reduce el riesgo de instalación, auditoría A8).

## 5. Catálogo de anomalías sintéticas (pre-registrado)

Se aplican sobre días reales del periodo evaluado; la posición y la hora se eligen con una semilla fija. Las horas evaluadas son 04–22 (ver la decisión sobre 23:00 en F2).

| Tipo | Escenario | Severa | Moderada | Leve |
|---|---|---|---|---|
| T1 | k horas consecutivas (k∈{1,2,3}) entre 05 y 21 multiplicadas por f | f=0,2 | f=0,5 | f=0,7 |
| T2 | Todo el día × f | f=0,6 | f=0,8 | f=0,9 |
| T3 | 2–3 horas de tarde o noche × f | f=2,0 | f=1,5 | f=1,25 |
| T4 | Error de datos: una hora ×2; una hora = 0; desfase de 1 h | todas se consideran severas | – | – |
| T5 | Perfil mediano de domingo o festivo (de la ventana de historia) en un día hábil | severa | – | – |

## 6. Fases, tareas, dependencias y criterios de aceptación

### F0 — Gobernanza, entorno y contraste independiente de la auditoría
Depende de: —

| ID | Tarea | Archivos | Criterio de aceptación / evidencia | Estado |
|---|---|---|---|---|
| F0.1 | Plan maestro y bitácora | `docs/PLAN_MAESTRO.md`, `docs/BITACORA.md` | Archivos en el repositorio | COMPLETADA (este commit) |
| F0.2 | Entorno reproducible: `requirements.txt` (runtime Linux), `requirements-dev.txt`, `requirements-research.txt`, `pyproject.toml` (paquete `src/`), `Makefile` | raíz | Instalación limpia en un venv 3.10 sin errores; `pytest` se ejecuta | PENDIENTE |
| F0.3 | Tests de contraste de la auditoría: reimplementación **independiente** que confirme o refute C1 (IQR sobre crudos), C2 (scaler con 912 filas), A1 (23:00 excluida), A2 (MSE en unidades incompatibles), C6 (demos obsoletos) | `tests/audit/test_audit_claims.py` | Cada afirmación queda como test que pasa (confirmada) o falla (refutada, se documenta) | PENDIENTE |
| F0.4 | CI básica (GitHub Actions: instalar y ejecutar pytest) | `.github/workflows/ci.yml` | Workflow en el repositorio; estado verificado si Actions está habilitado | PENDIENTE |

### F1 — Mitigación inmediata de seguridad (app legacy)
Depende de: F0.2

| ID | Tarea | Archivos | Criterio | Estado |
|---|---|---|---|---|
| F1.1 | Sustituir `pickle.loads` por un *unpickler* restringido (solo `numpy.ndarray`/`dtype`) y validar forma (20,), tipo numérico y valores finitos | `utils.py` | Test: la PoC de RCE ya no ejecuta comandos; los demos válidos siguen cargando; los NaN y las formas inválidas dan un error controlado | PENDIENTE |
| F1.2 | Retirar los 173 demos obsoletos (scaler distinto) | `dev/data_test/` | Test: el 100 % de los demos restantes coincide con una fila del dataset bajo el scaler desplegado | PENDIENTE |

*(La app se reescribe por completo en F5; F1 cierra la exposición mientras tanto.)*

### F2 — Capa de datos
Depende de: F0.2

| ID | Tarea | Criterio | Estado |
|---|---|---|---|
| F2.1 | `data.py`: carga de los 6 archivos, normalización de nombres de línea (mayúsculas y tildes), validación de esquema, tabla ancha por línea | Tests unitarios con fixtures pequeños y con los archivos reales | PENDIENTE |
| F2.2 | Duplicados entre archivos (2024-01-01 Línea B) y fechas faltantes (2020) | Regla determinista documentada; test | PENDIENTE |
| F2.3 | Semántica de NaN 22h/23h: análisis de datos + fuente oficial de horarios si se puede verificar | Decisión documentada con evidencia | PENDIENTE |
| F2.4 | Reporte de calidad de datos (por año y línea) | `reports/calidad_datos.md` generado por un script | PENDIENTE |
| F2.5 | `calendar.py`: festivos de Colombia, días especiales (24/12, 31/12), tipo de día; contraste con una fuente oficial para al menos un año | Tests; registro de la verificación | PENDIENTE |

### F3 — Problema, etiquetas y marco de evaluación
Depende de: F2

| ID | Tarea | Criterio | Estado |
|---|---|---|---|
| F3.1 | `labels.py`: heurística IQR corregida (residuos reales, incluye 23h) + análisis de calidad (variantes, kappa, sensibilidad a k, porcentaje explicado por calendario) | Informe `reports/etiquetas.md`; tests | PENDIENTE |
| F3.2 | `injection.py`: catálogo §5 con su verdad de terreno (día y horas afectadas) | Tests: magnitudes exactas, determinismo por semilla, sin modificar el original | PENDIENTE |
| F3.3 | Eventos reales: búsqueda en fuentes oficiales (Metro de Medellín, decretos nacionales o municipales) de fechas verificables en 2019–2024 | `data/eventos_reales.csv` con URL de la fuente; tarea BLOQUEADA si no hay fuentes verificables | PENDIENTE |
| F3.4 | `evaluation.py`: splits, *rolling origin* mensual, métricas (tasa de alertas, recall por escenario y severidad, recall horario, tasa en festivos), bootstrap por semanas, comparación pareada | Tests de las métricas con casos conocidos; test de no-fuga | PENDIENTE |
| F3.5 | `thresholds.py`: umbral por presupuesto de alertas, elegido en DEV | Tests | PENDIENTE |

### F4 — Modelado, selección y evaluación final
Depende de: F3

| ID | Tarea | Criterio | Estado |
|---|---|---|---|
| F4.1 | Detectores con interfaz común: naive, perfil robusto, regresión de calendario, AE original sin fuga, AE condicionado, IsolationForest | Test de causalidad (D8) para cada detector; test de determinismo | PENDIENTE |
| F4.2 | Selección en DEV (rolling origin): criterio primario = recall medio en escenarios moderados con un 5 % de alertas, sujeto a tasa en festivos ≤ 10 %; desempate (diferencia < 0,02 con IC solapados) → el modelo más simple o rápido | `reports/seleccion_dev.md` con IC; decisión en la bitácora | PENDIENTE |
| F4.3 | Congelar el modelo y el umbral; **una** evaluación en HOLDOUT (D1–D5) | `reports/holdout.md`; acceso registrado | PENDIENTE |
| F4.4 | Evaluación externa: 2019 (D6), 2020–2021 (estrés), eventos reales (D7) | `reports/externo.md` | PENDIENTE |
| F4.5 | Contexto multilínea (opcional): clasificar una alerta como sistémica (varias líneas) o local | Solo si F4.3 cumple; se evalúa con eventos reales y festivos | PENDIENTE |
| F4.6 | Artefacto final + metadatos (R3) | `artifacts/` | PENDIENTE |

### F5 — Aplicación
Depende de: F4.6

| ID | Tarea | Criterio | Estado |
|---|---|---|---|
| F5.1 | `service.py`: API pura (validar → puntuar → explicar) | Tests unitarios y de casos límite (fuzz) | PENDIENTE |
| F5.2 | Nueva `app.py` (fecha del histórico + carga de xlsx/csv); app antigua movida a `legacy/` | AppTest en verde | PENDIENTE |
| F5.3 | End-to-end con Playwright (subir archivo, elegir fecha, capturas) + latencia (A5) | Script e2e en verde; latencia medida | PENDIENTE |

### F6 — Reproducibilidad e infraestructura
Depende de: F4, F5

| ID | Tarea | Criterio | Estado |
|---|---|---|---|
| F6.1 | `make reproduce` de extremo a extremo + doble ejecución (D9) | Hashes o métricas idénticos | PENDIENTE |
| F6.2 | CI completa (tests + cobertura ≥ 80 %) | Workflow; resultado registrado | PENDIENTE |

### F7 — Documentación y cierre
Depende de: todo

| ID | Tarea | Criterio | Estado |
|---|---|---|---|
| F7.1 | README nuevo, ficha del modelo, ficha de los datos | Revisión de coherencia con los resultados | PENDIENTE |
| F7.2 | Informe final de validación (D1–D9, P, A, R) con pendientes explícitos | `docs/validacion/INFORME_FINAL.md` | PENDIENTE |

## 7. Riesgos y planes de contingencia
- **Ningún detector cumple D2 moderado.** Se informa; se analizan los escenarios que fallan; se evalúa el score por bloques de horas (CUSUM) antes de relajar un criterio, y cualquier relajación se documenta como desviación.
- **No hay eventos reales verificables.** D7 queda como limitación explícita; la validez externa se apoya en 2019–2021 y en la inyección.
- **El holdout contiene anomalías reales que inflan D1.** Se revisan manualmente las alertas del holdout y se informan con su explicación; no se "limpian" para mejorar la métrica.
- **El AE resulta superior.** Se conserva, con TF en el runtime y la justificación correspondiente.

## 8. Estado global
| Fase | Estado | Informe |
|---|---|---|
| F0 | EN PROGRESO | – |
| F1 | PENDIENTE | – |
| F2 | PENDIENTE | – |
| F3 | PENDIENTE | – |
| F4 | PENDIENTE | – |
| F5 | PENDIENTE | – |
| F6 | PENDIENTE | – |
| F7 | PENDIENTE | – |
