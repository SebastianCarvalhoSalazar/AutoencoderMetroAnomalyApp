# Bitácora de implementación

Registro cronológico de cambios, decisiones, hallazgos, pruebas, resultados y desviaciones. Las entradas se agregan, no se reescriben; si algo resulta incorrecto, una entrada posterior lo corrige.

Formato: `## AAAA-MM-DD — [Fase/Tarea] título` · **Cambio** · **Decisión** · **Evidencia** · **Resultado** · **Pendiente** · **Desviación**.

---

## 2026-11-23 — [F0.1] Creación del plan maestro y la bitácora
- **Cambio:** se crean `docs/PLAN_MAESTRO.md` y `docs/BITACORA.md`.
- **Decisión:** los criterios de terminado (D1–D9, P1–P4, A1–A5, R1–R5) y los splits (DEV 2022-01…2023-06, HOLDOUT 2023-07…2024-06, EXTERNO 2019/2020–2021) se fijan **antes** de implementar y de ver cualquier resultado del holdout.
- **Decisión:** las etiquetas heurísticas dejan de considerarse verdad de terreno. La validación primaria usa anomalías inyectadas con la verdad conocida por construcción, y eventos reales con fuente oficial cuando existan.
- **Evidencia previa:** auditoría `audit/AUDITORIA.md` (commit `c709546`), scripts en `audit/experiments/`.
- **Verificación de datos previa al plan:** los 6 archivos tienen el mismo esquema (Día, Línea de Servicio, 20 columnas horarias 04–23, Total general), sin negativos ni valores no numéricos, y sin duplicados (Día, Línea) dentro de cada archivo. El de 2023 incluye una fila del 2024-01-01 (Línea B) que también está en el de 2024.
- **Pendiente:** F0.2–F0.4.

## 2026-11-23 — [F0.2] Entorno reproducible
- **Cambio:** `requirements.txt` reescrito (runtime multiplataforma: sin `tensorflow-intel` ni `torch`; con `statsmodels`, `openpyxl` y `holidays`), más `requirements-dev.txt`, `requirements-research.txt` (TF 2.17/Keras 3.5), `requirements-lock.txt` (freeze completo), `pyproject.toml` (paquete `src/metro_anomaly`) y `Makefile`.
- **Decisión:** TensorFlow sale del runtime y queda solo en la capa de investigación. Su vuelta al runtime depende de que un AE resulte el modelo seleccionado en F4 (ver plan §7).
- **Evidencia:** venv limpio con Python 3.10.5 → `uv pip install -r requirements-dev.txt -e .` sin errores; importaciones verificadas (numpy 1.26.4, pandas 2.2.3, streamlit 1.39.0).
- **Pendiente:** verificar también con `pip` estándar (F6.1).

## 2026-11-23 — [F0.3] Contraste independiente de la auditoría
- **Cambio:** `tests/audit/test_audit_claims.py`. Reimplementa la verificación desde cero sobre `dev/results/*` (no reutiliza el código del notebook).
- **Resultado:** `pytest tests/audit` → **5 passed**. Quedan **confirmadas** C1 (etiquetas = IQR sobre crudos; con residuos reales cambian > 10 % de los días), C2 (scaler con 912 filas, min/max del dataset completo), C5 (> 90 % de festivos y de diciembre etiquetados), A1 (23h nunca evaluada) y A2 (MSE multiplicativo ≈ 1 frente al aditivo ≫).
- **Pendiente:** C6 y C7 se verifican (antes y después del cambio) en F1.

## 2026-11-23 — [F0.4] CI
- **Cambio:** `.github/workflows/ci.yml` (Python 3.10, `pytest -m "not research and not e2e"` con cobertura).
- **Pendiente:** confirmar la ejecución en GitHub Actions después del push.

## 2026-11-23 — [F1.1] Deserialización segura en la app legacy
- **Cambio:** `utils.py`: `leer_dato` usa un `Unpickler` restringido (solo `numpy` ndarray/dtype/_reconstruct) y valida la forma (20,), el tipo numérico y la finitud; la importación de TensorFlow pasa a ser diferida y `load_model` usa `compile=False` (elimina la advertencia de optimizador, auditoría M2). `app.py`: captura el `ValueError` y lo muestra con `st.error` en vez de una traza.
- **Decisión:** cambio mínimo, porque la app se reemplaza en F5; el objetivo es cerrar la RCE (C7) ya.
- **Evidencia:** antes, `tests/unit/test_legacy_utils.py` no se podía ni ejecutar (import de TF en un entorno sin TF) y la PoC de la auditoría ejecutaba comandos. Después: 29 passed; 389/389 demos cargan; E2E con Playwright: payload rechazado, archivo marcador no creado, 0 trazas.

## 2026-11-23 — [F1.2] Limpieza de demos obsoletos
- **Evidencia previa:** `tests/integration/test_demo_files.py` falló con 173 archivos que no coinciden con ningún día bajo el scaler desplegado (todos `x_test_normal_104…276`); se confirma C6.
- **Cambio:** `git rm` de esos 173 archivos (siguen recuperables en el historial de git).
- **Resultado:** el test pasa (216/216 archivos exactos y con la etiqueta correcta). Suite: 35 passed.

## 2026-11-23 — [F2] Capa de datos y calendario
- **Cambio:** `src/metro_anomaly/data.py` (carga, validación de esquema, normalización de líneas, duplicados entre archivos, tabla diaria sin imputación, reporte de calidad), `src/metro_anomaly/calendar.py` (festivos, tipo de día, rasgos de calendario), `scripts/data_quality.py`, `reports/calidad_datos.md`; tests `tests/unit/test_data.py`, `tests/unit/test_calendar.py`, `tests/integration/test_real_data.py`; `tabulate` añadido a requirements.
- **Hallazgo nuevo:** 18 días de 2019 y 2021 con horas faltantes en 04–21. El primer borrador del test suponía 04–21 completos y **falló** (35 NaN). Se investigó en lugar de relajar el test, y ahora el test caracteriza el hecho exacto.
- **Hallazgo nuevo:** el duplicado 2024-01-01 Línea B tiene valores **distintos** entre archivos.
- **Decisión:** 23h excluida de la puntuación; sin imputación; horas faltantes → alerta de calidad de datos.
- **Bloqueo:** la verificación oficial de horarios (metrodemedellin.gov.co) y del texto de la Ley 51/1983 (procuraduria.gov.co, funcionpublica.gov.co) no fue posible: DNS bloqueado en el entorno. Se sustituye por un contraste empírico (40/40) y se registra como limitación.
- **Desviación:** F2.5 queda COMPLETADA PARCIAL (sin fuente oficial).
- **Resultado:** `pytest` → 62 passed; cobertura 97 % en los módulos nuevos.

## 2026-11-23 — [F3 + F4.1] Marco de evaluación, etiquetas, inyección y detectores
- **Cambio:** `labels.py`, `injection.py`, `evaluation.py`, `detectors.py`, `detectors_nn.py`, `data/eventos_reales.csv`, `scripts/label_quality.py`, `reports/etiquetas.md`, más tests de unidad para cada módulo.
- **Decisión de diseño:** todos los detectores de forma trabajan en log1p menos un **nivel causal** (mediana de log Σhoras de los días hábiles y sábados de [t-28, t-1]). Así se adaptan a cambios de demanda (A3) sin usar el futuro. El AE original se reproduce **sin** nivel, como referencia fiel.
- **Decisión:** el umbral = cuantil 95 % de los scores fuera de muestra de los días reales de DEV (presupuesto de alertas del 5 %); se congela antes del holdout.
- **Hallazgo:** las etiquetas heurísticas tienen un acuerdo moderado entre variantes razonables (κ 0,48–0,54) y marcan el 95,5 % de los festivos en todas. Refuerza que no son verdad de terreno.
- **Eventos reales:** 2 eventos oficiales (EV1, EV2), ambos de 2020. No hay ninguno en 2022–2024 → D7 queda como limitación (criterio pre-registrado).
- **Tests modificados:** ver la sección "Cambios a tests" de `docs/validacion/F3.md` (un test con una hipótesis errónea, reemplazado por el mecanismo correcto; ajuste de las etiquetas ante 2 celdas NaN).
- **Resultado:** 99 passed + 1 skipped (sin TF) + 4 passed (TF). (Corrección: una primera versión de esta entrada decía "103 passed" por error de transcripción.)

## 2026-11-23 — [F4.2] Selección DEV ronda 1: RESULTADO NEGATIVO y reevaluación
- **Resultado (`reports/seleccion_dev.md`, ronda 1):** con un presupuesto de alertas del 5 %, el umbral de todos los detectores de z está en |z| ≈ 10, y el recall moderado medio va de 0,17 a 0,48 (la meta D2 es 0,80). La regla eligió `perfil_robusto/max` (empate con `ae_original`: 0,340 frente a 0,346). `ae_contextual` tiene el mejor recall en T1 (0,94), pero **no detecta T5** (0–0,02) y viola la restricción de festivos (18–21 %).
- **Diagnóstico (solo DEV):** la cola de los días reales la generan efectos de calendario no modelados: jornadas electorales (29/05/2022 y 19/06/2022, z ≈ −55), Sábado Santo (16/04/2022, 08/04/2023) y la temporada del 24/12 al 06/01 (decenas de días con caídas de 4 a 7σ). Además hay caídas aisladas de una hora a las 16h en días hábiles (2022-08-05, 2022-05-19, 2022-11-15, 2023-03-30), posibles incidentes reales **sin fuente**.
- **Decisión (desviación del plan, documentada):** se hace **una única ronda 2** de desarrollo en DEV antes de congelar. Los criterios D1–D9 **no se modifican** y el holdout sigue sellado. Cambios pre-declarados:
  - R2.1: el calendario incorpora la Semana Santa (lunes a sábado; Pascua calculada) y una temporada de fin de año (20/12–10/01, ya existente como rasgo) como modificadores del tipo de día. Las jornadas electorales (fechas oficiales) **no** son contexto: se dejan como anomalías que deben alertarse y se anotan en la explicación.
  - R2.2: `perfil_contextual` = perfil con tipos de día extendidos (hábil/sábado en temporada) y un nivel que excluye los días de temporada.
  - R2.3: `regresion_contextual` = ridge con rasgos de Semana Santa y sigma por (tipo de día × hora).
  - R2.4: agregación `peak_shift` = máximo entre el pico estandarizado (max|z|) y el desplazamiento global estandarizado (|media z|), con escalas robustas estimadas en la historia de entrenamiento (causal), para captar T2.
- **Eventos reales:** se registran EV3 (elecciones del 27/10/2019) y EV4 (29/10/2023) como **prospectivos**, antes de ver los scores de 2019 y del holdout. EV5 y EV6 (elecciones de 2022) se descubrieron a posteriori en DEV y **no cuentan** para D7. EV7 (congreso 2022, fuente UE) queda en DEV, también fuera de D7. Fuentes: registraduria.gov.co (resultados de búsqueda; acceso directo no disponible).
- **Riesgo reconocido:** iterar en DEV aumenta el riesgo de sobreajuste a DEV; el holdout, que se abre una sola vez, es el árbitro.

## 2026-11-23 — [F4.2] Selección DEV ronda 2 y CONGELAMIENTO (antes de abrir el holdout)
- **Verificación de no-regresión:** con el código de la ronda 2, `perfil_robusto/max`, `regresion_calendario/max` y `naive_semanal/max` reproducen **bit a bit** los scores de la ronda 1 (reales e inyectados).
- **Test corregido:** `test_peak_shift_detects_global_shift_better_than_max` usaba T2_leve, que en la serie sintética satura ambas agregaciones (recall 1,0 en las dos). Se cambió a ×0,95; observado: max 0,75 frente a peak_shift 1,00.
- **Resultado ronda 2 (`reports/seleccion_dev.md`; la ronda 1 se conserva en `reports/seleccion_dev_ronda1.md`):** elegida **`perfil_contextual/top2`**, umbral **7,3482** (cuantil 95 % de DEV). Recall moderado medio 0,578 [IC 0,532–0,621]; T1/T2/T3 moderados 0,74/0,58/0,42; severas 0,867; T5 0,964; festivos 8,3 %; localización 0,966. Es la mejor por el criterio primario, sin empates.
- **Expectativa explícita:** el resultado de DEV **no alcanza** D2 (moderada 0,80; severa 0,95). Según el plan no hay más rondas: el holdout medirá el desempeño real y D2 se informará como incumplida si así resulta.
- **Hallazgos de la ronda 2:** (a) la regresión contextual baja la tasa en festivos a 0 % pero tiene menor recall; (b) `ae_contextual` sigue sin detectar T5 (≤ 0,02) y viola la restricción de festivos; se descarta para producción; (c) T4_duplicada (una hora ×2) es el escenario severo más débil (0,60 con el elegido).
- **Congelado:** `configs/seleccion.yaml` (detector, agregación, umbral y umbrales de todas las líneas base). Commit previo a cualquier ejecución sobre el holdout.

## 2026-11-23 — [F4.3] Holdout abierto (1.ª vez) con el modelo congelado en `7ad6527`
- **Ejecución:** `scripts/run_eval.py --period holdout` para las 18 configuraciones; `scripts/evaluate_holdout.py`. Resultado del modelo elegido (`perfil_contextual/top2`, umbral 7,3482), **tal como se ejecutó**: D1 0,025 ✔ · D2 severas 0,877 ✘ · D2 moderadas 0,603 ✘ · D3 0,111 ✘ (2/18: Navidad, Año Nuevo) · D4 0,955 ✔ · D5a ✔ (mejor que todas las líneas base, IC > 0) · D5b ✔ (+0,303 frente al AE original, IC 0,25–0,35).
- **Externo (con el mismo código):** D6 2019 = 0,109 ✔ (AE original 0,167 ✘); D7 = 4/4 eventos prospectivos detectados (todos los detectores los detectan; poco discriminante); COVID: el perfil alerta ~100 % de abril a junio de 2020 y tarda unos 5 meses en readaptarse (limitación ante cambios de régimen).

## 2026-11-23 — [F4.3] HALLAZGO: defecto de implementación en los detectores de la ronda 2 (descubierto después de abrir el holdout)
- **Defecto:** con `exclude_season=True`, el nivel causal de [t-28, t-1] queda con menos de 10 días válidos a principios de enero (la temporada va del 20/12 al 10/01) → nivel NaN → score NaN. Afecta a `perfil_contextual` y `regresion_contextual`: 17 días de DEV (2023-01-05…), 18 del holdout (2024-01-05…) y 32 del periodo COVID.
- **Defecto de medición asociado:** `metrics()` trataba un score NaN como "sin alerta" (`NaN > umbral` = False) y como inyección no detectada. Por eso **los resultados de la ronda 2 y del holdout anteriores están sesgados**: la tasa de alertas a la baja; el recall, en parte a la baja por los NaN.
- **Cómo se detectó:** un diagnóstico post-hoc (`np.quantile` devolvió NaN). Los tests no lo cubrían: la serie sintética de los tests no tiene una temporada de ~3 semanas que vacíe la ventana.
- **Decisión:**
  1. Corrección mínima y determinista, sin parámetros ajustables: si la ventana de 28 días no alcanza el mínimo de días válidos, se amplía de forma causal a 56 y luego a 91 días. No se usa ningún dato del holdout para decidirla.
  2. `metrics()` reporta la **cobertura** (fracción de días puntuados) y un test exige cobertura = 100 % en los días completos.
  3. Se **repite la selección DEV con la misma regla**, sin otros cambios, y se re-evalúa el holdout. Se publican las dos versiones (con y sin el defecto) para que la desviación sea trazable.
- **Riesgo reconocido:** es una segunda lectura del holdout. Se justifica porque se corrige un defecto de implementación, no un ajuste de modelado; el umbral y la configuración salen de DEV con la regla pre-registrada.

## 2026-11-23 — [F4.3–F4.6] Resultado final y artefacto
- **Corrección aplicada:** respaldo causal del nivel (56/91 días) + `metrics()` con cobertura y exclusión de NaN. El test `test_full_coverage_through_year_end_season` falla con el código anterior (2 detectores) y pasa con el corregido. Los detectores no afectados reproducen bit a bit DEV, holdout y COVID.
- **Selección repetida (misma regla):** con la corrección, `perfil_contextual/top2` baja en DEV de 0,578 a 0,395. **La mejora de la ronda 2 estaba inflada por el defecto.** Elegido `perfil_contextual/peak_shift`, umbral 7,388.
- **Holdout (2.ª lectura):** D1 0,022 ✔ · D2 severas 0,894 ✘ · moderadas 0,506 ✘ · D3 0,111 ✘ · D4 1,00 ✔ · D5a ✔ (no inferior) · D5b ✔ (+0,207). **Externo:** D6 0,075 ✔ · D7 4/4 ✔.
- **Corrección de presentación:** la tabla de alertas del holdout comparaba |z| por hora con el umbral del score agregado (escalas distintas); ahora lista las 4 horas de mayor |z| y el z medio.
- **Artefacto:** `artifacts/modelo.json` (perfil_contextual/peak_shift, entrenado con 2023-07-02…2024-06-30, umbral de DEV, hash SHA-256 de los 6 archivos, commit, métricas del holdout).
- **Decisión:** se despliega exactamente la configuración evaluada (umbral de DEV), aunque en el holdout resultó conservadora (2,2 % de alertas). La recalibración periódica queda como trabajo futuro no evaluado.
- **Estado:** el modelo es funcional y supera claramente al sistema original, pero **no cumple todos los criterios pre-registrados (D2, D3)**. Se informa así; no se declara éxito del modelo.

## 2026-11-23 — [F5] Aplicación nueva
- **Cambio:** `service.py`, `app.py` nuevo, app original en `legacy/`, tests `test_service.py` (18), `test_app.py` (4) y E2E (`tests/e2e/`).
- **Resultado:** todo en verde; E2E p95 de 665 ms; la interrupción inyectada se detecta vía archivo; el archivo inválido da un error controlado.
- **Incidencias:** (1) selector E2E débil que dio una captura en estado de carga → corregido; (2) un test mal nombrado → renombrado. Detalle en `docs/validacion/F5.md`.
- **CI:** GitHub Actions activo; ejecuciones 2–5 en `success` (verificado con la API de Actions).

## 2026-11-23 — [F6] Reproducibilidad
- **Cambio:** `scripts/reproduce.py`, `scripts/check_determinism.py`, objetivos de Make (`reproduce`, `reproduce-research`, `determinism`, `e2e`), CI con `--cov-fail-under=80` y un job `research` con TensorFlow.
- **Resultado D9:** 8/8 salidas con SHA-256 idéntico entre dos ejecuciones aisladas (`reports/determinismo.md`). La copia limpia sin TF reproduce exactamente la selección, el umbral y las métricas de las 15 configuraciones sin AE versionadas.
- **Hallazgo menor:** `artifacts/modelo.json` registraba `git_commit_build = 7ad6527` porque se construyó antes de hacer commit de la corrección del nivel; el contenido era correcto (lo confirma la comparación de hashes). Se reconstruye desde un commit limpio.

## 2026-11-23 — [F7] Documentación y cierre
- **Cambio:** `README.md` reescrito, `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`, `docs/validacion/INFORME_FINAL.md`, `legacy/README.md`.
- **Corrección antes de publicar:** el primer borrador de la ficha del modelo citaba recall por escenario de `perfil_contextual/top2` (configuración anterior). Se reemplazó por las cifras del modelo final, tomadas de `reports/holdout_metricas.csv` (T3 moderado 0,253; T1 moderado 0,745; T2 moderado 0,522).
- **Veredicto:** sistema funcional, reproducible y validado; el modelo incumple D2 y D3 (informado sin relajar criterios).

## 2026-11-23 — [F6.2] CI: el job `research` falló en GitHub Actions (ejecución 8)
- **Evidencia:** `test_causality_and_severe_detection[ae_original]`: 484,56815 frente a 484,56882 (dif. relativa 1,4e-6 > tolerancia 1e-6). En local pasaba.
- **Diagnóstico:** no es una fuga. El AE original no lee el contexto; la diferencia viene de comparar la predicción de un lote de 28 días con la de un día individual (float32 en TensorFlow/oneDNN depende del tamaño del lote y de la CPU).
- **Corrección del test:** la causalidad se verifica con igualdad exacta y el mismo tamaño de lote (con y sin el futuro alterado); la comparación lote frente a día individual se mantiene aparte, con tolerancia de float32 (rel 1e-4), documentada en el test.
- **Lección:** los tests de equivalencia numérica con redes neuronales deben separar el efecto que se prueba (causalidad) de la variabilidad de punto flotante.
- **Verificación:** la ejecución 9 de GitHub Actions (commit `9e1b821`) terminó en `success` con los dos jobs (`tests` con cobertura ≥ 80 % y `research` con TensorFlow).

## 2026-11-23 — Cierre
- Todas las fases F0–F7 están COMPLETADAS, salvo las limitaciones documentadas (F2.5 sin fuente oficial; F4.5 no ejecutada porque no se cumplió su condición previa). El modelo **incumple D2 y D3**: se informa en README, ficha del modelo e informe final.
