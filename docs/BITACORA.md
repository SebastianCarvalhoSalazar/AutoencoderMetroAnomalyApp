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

## 2026-11-23 — [F8.1] Fase 2: validación de preparación para producción y de UX (planificación)
- **Solicitud:** validación end-to-end de producción (simulación cronológica, latencia, robustez, eventos reales en fuentes oficiales colombianas, protocolo de validación humana) y validación funcional, visual y de UX con navegador real.
- **Decisión:** se pre-registran los criterios PR1–PR8 y UX1–UX7 en `docs/PLAN_MAESTRO.md` §9 antes de ejecutar cualquier prueba.
- **Regla de fuentes:** los eventos se validan solo con plataformas oficiales colombianas (dominios `.gov.co` y el sitio del Metro). Todo contenido web se trata como dato no confiable y se revisa en busca de instrucciones incrustadas (prompt injection) antes de usarlo.

## 2026-11-23 — [F9.1] Línea base de UX (antes de cambios)
- **Ejecución:** `tests/e2e/ux_audit.js` sobre la app de F5, con 17 archivos de prueba (`tests/e2e/make_ux_fixtures.py`) y axe-core 4.10.2. Capturas en `docs/validacion/img/ux/antes/`.
- **Hallazgo en las pruebas (T-01):** el primer intento mostró el mismo resultado en todas las fechas. Causa: `fill()` no aplica la fecha en el campo con máscara de Streamlit. Consecuencia: **la latencia publicada en F5 (p95 665 ms) medía reejecuciones sin cambio de fecha y no es evidencia válida**. Se corrigieron los dos scripts (escritura carácter a carácter + verificación).
- **Resultado de la línea base:** 0 errores de JS, excepciones y peticiones fallidas; 16 defectos de UX documentados (comunicación de alertas, leyenda, banda, formato, accesibilidad, límite de carga, etc.).

## 2026-11-23 — [F8.2–F8.5] Simulación cronológica y escenarios
- **Cambio:** `src/metro_anomaly/simulation.py`, `scripts/production_validation.py` → `reports/produccion/validacion_produccion.json`.
- **Resultados:** PR1 0 violaciones (idéntico a F4); PR2 0,04 s por día; PR5 99,7 %; intra-día con retardo mediano de 1 h y 3,6 % de falsas alarmas parciales; escalón −20 % absorbido en pocos días; recurrentes alertan siempre; estacional −15 % casi invisible; precisión en flujo mixto 0,35 (prevalencia 2 %) / 0,59 (5 %), con FP dominados por las 8 alertas reales sin etiqueta.
- **Defecto corregido:** una línea sin días completos (Línea L) fallaba con un error opaco al puntuar → error explícito al ajustar (test de regresión).

## 2026-11-23 — [F8.6] Eventos reales en fuentes oficiales
- **Bloqueo y solución:** `telemedellin.tv` y `www.metrodemedellin.gov.co` no resolvían (DNS). Se propuso un entorno personal con solo 3 hosts oficiales (`telemedellin.tv`, `www.metrodemedellin.gov.co`, `www.medellin.gov.co`); el usuario lo aprobó.
- **Control de prompt injection:** páginas descargadas a un directorio aislado y texto extraído con `python -I`. Coincidencias solo en código del sitio (WP Rocket, comentarios de plantilla, cookies, menú "Asistente virtual"). **Sin instrucciones incrustadas.**
- **Resultados:** IN1 (05/02/2023, operación parcial) sin señal; IN2 (22/08/2023, 4 estaciones cerradas desde las 07:05) sin señal; IN3 (28/08/2023, suspensión total en hora pico) detectado. Con dato horario habría alertado al cierre de las 17 h, después del aviso oficial de las 17:30. **Sensibilidad a incidentes reales: 1/3.**
- **Hallazgo:** jornadas electorales = **no cobro del pasaje** (comunicado oficial; desde 2006; el 29/10/2023 hasta las 17:00) → entradas ≈ 0 hasta las 16 h en todas las líneas. Las detecciones electorales de D7 son un **efecto de medición**. Se creó la categoría "Efecto de medición conocido" y se corrigió la interpretación de D7 en F4, el informe final y la ficha del modelo.

## 2026-11-23 — [F8.7] Operación, auditoría y monitoreo
- **Cambio:** `src/metro_anomaly/operations.py`, `scripts/run_daily.py`, `tests/integration/test_operations.py` (7).
- **Ajustes motivados por evidencia:**
  - Regla de valor implausible: la versión "> 5× lo esperado" marcaba el 08/12/2023 22 h, un patrón real de temporada dentro del rango histórico. Se cambió a "> 1,5 × máximo histórico de la hora". Esta regla **sí** detecta los conteos acumulados de 2019 (11 días, 10 en 2019).
  - Aviso de monitoreo: "90 días sin alertas" tenía P ≈ 13 % de dispararse en falso con la tasa observada; pasó a 180 días (≈ 2 %).
- **Severidad empírica:** el artefacto incluye los 456 scores de referencia de DEV (p95 = umbral exacto, p99 15,09, máx 43,57).

## 2026-11-23 — [F9.2–F9.4] Correcciones de UX y nueva verificación
- **Cambios en la app:** categorías honestas de alerta, severidad y frecuencia histórica, aviso de "no es un incidente confirmado", categoría de calidad de datos y de efecto de medición, leyenda/unidades/formato es-CO/horas sin dato/recorte de la banda, línea de tiempo del score, tabla accesible, descarga CSV, fechas dd/mm/aaaa, validación del rango de fechas, `.streamlit/config.toml` (20 MB, sin opciones de desarrollo, sin trazas, contraste AA).
- **Incidencias en las pruebas:** (T-02) la máscara del campo de fecha conservaba un dígito y los pasos H11–H13 mostraban resultados obsoletos; se detectó con la bandera `applied` y se corrigió. El verificador UX3 marcaba como fallo la tabla dentro de un expansor cerrado y las horas en días de problema de datos; se corrigió para tratarlos como "no aplicable" sin relajar las demás comparaciones.
- **Iteraciones visuales:** la leyenda en móvil requirió 3 iteraciones (3 columnas → 2 columnas con nombres largos → 2 columnas con nombres cortos) hasta quedar completa (`despues/M03`).
- **Resultado final:** 35 pasos con 0/0/0/0; UX3 100 % (11 vistas + 3 CSV idénticos); axe con 0 violaciones atribuibles a la app; sin desbordamiento a 390 px; E2E p95 669–685 ms con cambio verificado.

## 2026-11-23 — [F9.5] Dictamen y cierre de la fase 2
- **Pruebas:** 160 en total (154 rápidas, 4 de autoencoders, 2 E2E con navegador), todas en verde; cobertura del 82 %.

  | Archivo de pruebas | Cantidad |
  |---|---|
  | unit/legacy_utils | 29 |
  | unit/detectors | 33 |
  | unit/data | 16 |
  | unit/injection | 10 |
  | unit/calendar | 7 |
  | unit/evaluation | 6 |
  | unit/labels | 4 |
  | unit/artifacts | 3 |
  | unit/detectors_nn | 4 |
  | integration/service | 21 |
  | integration/operations | 7 |
  | integration/app | 6 |
  | integration/real_data | 6 |
  | integration/demo_files | 1 |
  | audit | 5 |
  | e2e | 2 |
- **Dictamen:** NO APTO como detección temprana de incidentes (bloqueantes B1–B3); APTO CON CONDICIONES como apoyo de revisión retrospectiva y control de calidad de datos (`docs/validacion/DICTAMEN_PRODUCCION.md`).

## 2026-11-23 — [F9.5] Verificaciones finales
- El artefacto se reconstruyó desde un commit limpio (`git_commit_build = d40e14e`), corrigiendo otra vez el desfase de procedencia: se había construido antes de hacer commit del código de los scores de referencia.
- `reports/determinismo.md` se regeneró con el código actual: 8/8 salidas idénticas. El SHA-256 normalizado del artefacto versionado (`ec9cf22f0678e164`) coincide con el de las dos ejecuciones aisladas.
- CI de GitHub Actions: ejecuciones 12, 13 y 14 en `success`.

## 2026-11-24 — Guía visual y corrección de un defecto declarado como resuelto
- **Cambio:** `tests/e2e/guia_visual.js` genera 7 capturas anotadas de los flujos principales (`docs/guia_visual/`).
- **Hallazgo:** el defecto V2 (eje del score de la línea de tiempo con solo 0 y 50), registrado como corregido en F9, **no lo estaba**. Causa: las capas del umbral usaban un eje por defecto que, al fusionarse con el de las demás, anulaba las marcas configuradas. Un primer intento (`axis=None` en esas capas) eliminó el eje completo. Corrección: todas las capas comparten la misma escala y el mismo eje. Verificado visualmente (`docs/guia_visual/02_*`, `03_*`: marcas 0, 1, 2, 5, 10 y 50).
- **Pruebas:** suite rápida en verde.

## 2026-11-24 — [F10] Fase 3: plataforma multilínea (planificación)
- **Decisión:** se pre-registran las hipótesis H1–H7 con sus criterios en `docs/PLAN_MAESTRO.md` §10, antes de cualquier experimento.

## 2026-11-24 — [F10.1] Inventario y EDA multilínea
- **Artefactos:** `scripts/eda_multilinea.py` → `reports/fase3/eda_multilinea.{md,json}`; `data/lineas.csv` (modo, extremos, km y estaciones por línea) y `data/topologia_red.csv` (conexiones entre líneas en estaciones de integración). Ambos tienen una URL oficial del Metro por fila; las páginas descargadas se revisaron en busca de instrucciones inyectadas (no se hallaron).
- **Hallazgos:**
  - En 2023, la Línea A concentra el 63,7 % de las entradas registradas, seguida de la Línea 1 con 10,5 %, la B con 9,5 % y la T-A con 5,45 %; las demás líneas tienen menos del 2,4 %.
  - El primer componente principal del residuo diario desestacionalizado explica el 87 % de la varianza: las líneas se mueven juntas. La Línea P es la menos correlacionada (≈ 0,6 con el resto).
  - Hay 41 días con ≥ 3 líneas sin dato a la misma hora: 19 en 2019, 9 en 2020, 7 en 2021, 2 en 2022 y 4 en 2023.
  - El 28/08/2023 (IN3), la Línea A cae con z de −13,6, −61 y −137 entre las 17 y las 19 h, mientras que la O y la 2 suben.
  - La Línea L se excluye del modelado por hora: no opera en todas las franjas, así que no tiene días completos.
- **Defecto propio:** un `KeyError` en líneas cuya serie termina antes del 30/06/2024; se corrigió con `reindex`.

## 2026-11-24 — [F10.2] Fuentes externas
- **Decisión:** la alineabilidad se evaluó solo con portales oficiales (`docs/fase3/FUENTES_EXTERNAS.md`).
  - Se descartan para modelos línea × hora (H7): proyecciones de población por comuna (anuales y solo de Medellín), la Encuesta Origen-Destino (estática y sin microdatos públicos), la amenaza por movimientos en masa del POT (estática) y los corredores del POT (dominio de `tipo_sistema` no publicado). **H7 se descarta**; esas fuentes quedan para una fase con datos por estación.
  - Se integra la lluvia horaria de Open-Meteo (reanálisis, CC BY 4.0) con `data/clima_open_meteo_medellin.csv`, sus metadatos y `src/metro_anomaly/fuentes/clima.py`. Alineación: la precipitación es la suma de la hora precedente, así que la franja h toma el registro h+1 (cubierto por una prueba).
- **Cambio de entorno (aprobado por el usuario):** se permitió el host `archive-api.open-meteo.com`.
- **Limitación:** una sola celda de reanálisis para todo el valle. Las series de pluviómetros del SIATA no tienen descarga pública.

## 2026-11-24 — [F10.3] H1–H4: experimentos de red (`scripts/fase3/exp_red.py` → `reports/fase3/red.{md,json}`)
- **H1 — detector de coherencia entre líneas** (`src/metro_anomaly/red.py::CoherenceDetector`): residuo del z de la línea frente a la mediana de z de las otras 10 líneas en la misma hora, con un ajuste Huber por hora. Mismo presupuesto de alertas (5 % de días reales) que `perfil_contextual/peak_shift`.

  | Línea, periodo | Recall moderado (perfil → coherencia) | Diferencia, IC 95 % | Alertas en festivos (perfil → coherencia) |
  |---|---|---|---|
  | A, holdout | 0,702 → 0,900 | +0,198 [0,177; 0,216] | 0,222 → 0,111 |
  | A, DEV | — | +0,253 [0,207; 0,293] | — |
  | B, holdout | — | +0,052 [0,026; 0,077] | — |
  | 1, holdout | — | +0,091 [0,067; 0,110] | — |
  | T-A, holdout | — | −0,001 [−0,011; 0,010] | — |

  - **Criterio pre-registrado (Línea A): CUMPLE.**
  - **Límites:** las inyecciones son locales por construcción, que es justo el caso que este detector favorece. En días reales del holdout, la coherencia deja de alertar el 25/12/2023 y el 29/03/2024 (Viernes Santo), ambos sistémicos, y añade 31/10, 16/11, 15/12/2023, 20/01 y 30/03/2024. Mantiene IN3 (score 127,9; umbral 6,37) y las elecciones.
  - **Ni este detector ni el anterior** ven IN1 (05/02/2023) ni IN2 (22/08/2023), con scores 1,0 y 0,5: las interrupciones parciales siguen siendo invisibles con datos línea × hora.
- **H2 — alcance local / multilínea / generalizado** (`classify_scope`; umbrales fijados antes de evaluar: |z| > 3, fracciones 0,2 y 0,5): exactitud con inyecciones de **0,884 < 0,90**. **NO CUMPLE.**
  - La mayoría de los errores caen en la clase intermedia "multilínea".
  - Las elecciones del 29/10/2023 salen "generalizada" y el 28/08/2023 sale "multilínea" (fracción 0,40: 1, B, M, T-A).
  - **Advertencia:** la lectura de "multilínea" para el 28/08 como "no local" se fijó después de ver los z del EDA, así que no es una prueba ciega.
  - Uso permitido: solo como **indicador descriptivo** con su exactitud declarada, no como clasificador validado.
- **H3 — redistribución el 28/08/2023, 17–19 h:** la Línea 2 sube con z = 6,8 a las 19 h (perfil contextual por línea), lo que **cumple** el criterio descriptivo.
  - La Línea O sube con z = 11,1 frente al mismo día de las 8 semanas previas (EDA), pero solo con z = 2,1 frente al perfil contextual: la señal **depende del método de referencia**.
  - Con n = 1 no se puede afirmar un patrón de redistribución ni una causalidad.
- **H4 — ausencia simultánea de registro en ≥ 3 líneas habituales:**
  - Recall de 1,0 en 200 fallas simuladas en 3–6 líneas, y alertas en 4 de 731 días reales (0,55 %). **CUMPLE** los criterios numéricos.
  - **Hallazgo que cambia la interpretación:** ninguno de los 4 días reales es una falla de recaudo.
    - 28/08/2023, 19 h: la suspensión IN3.
    - 29/10/2023: las elecciones, sin cobro en metro y cables, es decir, un efecto de medición.
    - 24 y 31/12/2023, 22 h: un cierre anticipado que se repite en los datos de 2019, 2022 y 2023. No se verificó el comunicado oficial de 2023.
  - Por eso la regla se adopta como "ausencia simultánea de registro: causa por determinar (operación, calendario o datos)", **no** como "falla de recaudo".
  - El 3,5 % de "alertas con 1–2 líneas simuladas" se debe a celdas que ya tenían 1–2 líneas reales sin dato (5 % de las celdas); la regla cuenta el total, como fue diseñada.

## 2026-11-24 — [F10.3] H5, H5b y H6: pronóstico del día siguiente (`scripts/fase3/exp_pronostico.py`)
- **Primera ejecución (v1, holdout 2023-07…2024-06).** Modelos: GBM global con rezago, nivel causal y calendario, más regresión cuantílica p10/p90.
  - Un primer cálculo comparaba modelos sobre filas distintas: el naive no tiene referencia en días incompletos de los cables. Se **corrigió** para usar solo las filas comunes (69 252 de 74 080).
  - **H5:**
    - El WAPE global baja de 0,0725 con el naive a 0,0564 con el GBM. En la Línea A baja de 0,0682 a 0,0517 (−24 %).
    - El GBM mejora al naive en ≥ 10 % en 9 de 11 líneas; las excepciones son H (9,6 %) y P (8,2 %).
    - La cobertura del intervalo del 80 % de la Línea A es **0,723 < 0,75: NO CUMPLE**. Las demás líneas están entre 0,758 y 0,803.
    - En los cables H, K y M, el perfil contextual supera al GBM.
  - **H5b — lluvia, como oráculo con la lluvia observada:** empeora el WAPE en las 11 líneas (−0,1 % a −3,7 %). **SE DESCARTA.**
    - La comparación "días lluviosos" está confundida con el mes: 16 de los 37 días lluviosos caen en abril–mayo de 2024, justo después de Semana Santa, cuando el rezago del GBM toma días de temporada.
  - **H6:**
    - El AUC del GBM es > 0,80 y supera al naive en A (0,971 frente a 0,942), 2, J, M, O, P y T-A.
    - **No cumple** en B (0,927 frente a 0,935) ni en K (0,844 frente a 0,859), ni en H (0,711).
    - La Línea 1 solo tiene 3 positivos.
- **Ajustes posteriores a ver el holdout v1** (declarados aquí antes de su ejecución; la evidencia que aportan es más débil):
  1. **Intervalo:** calibración conformal en línea y causal, con los meses anteriores de cada línea.
  2. **Variante `gbm_v2`:** rezago por tipo de día extendido, para que un día normal no tome como referencia un día de temporada.
  3. **Combinaciones `combo` y `combo_v2`:** media en escala log de GBM y perfil.
  4. **Regla de selección (pre-registrada ahora):** para cada línea se elige el candidato con menor WAPE en DEV (2022-07…2023-06), entre `perfil`, `gbm`, `gbm_v2`, `combo` y `combo_v2`. Luego se informa el holdout del híbrido elegido sin modificarlo. Los candidatos `perfil` y `combo` se añadieron por lo observado en v1, lo que se declara como sesgo de selección de candidatos.

## 2026-11-24 — [F10.4] Arquitectura de la plataforma y defecto hallado por las pruebas
- **Cambio:** `src/metro_anomaly/plataforma.py` orquesta las fuentes, los modelos y las reglas, y escribe un manifiesto de procedencia. Lo complementan `scripts/fase3/build_plataforma.py`, que genera los artefactos en `artifacts/plataforma/`, y `tests/unit/test_plataforma.py`, que prueba el motor de reglas sobre una red sintética.
  - Reproducción cronológica mensual causal sobre 11 líneas: perfil contextual y coherencia (A, B, 1).
  - Umbral por línea = p95 de su score en DEV.
  - Prioridad de reglas: efecto de medición (elecciones, solo metro, tranvía y cable, según el comunicado oficial) > ausencia simultánea en la red > problema de datos (solo en horas que la línea suele reportar) > anomalía con alcance orientativo > sin anomalía. Una categoría "no evaluable" cubre los días incompletos en horas que la línea no siempre reporta.
- **Defecto (propio):** en el índice de red, un **cero** en otra línea (falla de datos) producía un z muy negativo que arrastraba la mediana. El resultado era una "anomalía local" falsa en una línea sana; la prueba `test_simultaneous_absence_and_single_line_data_problem` lo detectó en la Línea 1.
  - **Corrección:** los ceros se tratan como ausencia de registro en `network_z` y en el clasificador de alcance.
  - Como esto cambia el detector evaluado en H1, **se re-ejecuta `exp_red.py`**. La versión anterior del informe se conserva en el historial de la bitácora (entrada H1–H4).

## 2026-11-24 — [F10.3] Re-ejecución de H1–H4 y resultado final de H5/H6
- **H1–H4 tras corregir el tratamiento de los ceros:** `reports/fase3/red.md` es idéntico a la versión anterior. En las 11 líneas principales no hubo ceros que afectaran el índice de red en los periodos evaluados.
- **H5, DEV (2022-07…2023-06):** el GBM solo es **peor** que el naive en varias líneas (A: −14,5 %; K: −18,2 %; P: −20,9 %), porque en los primeros meses solo había 6 meses de historia. `combo_v2` (media de `gbm_v2` y perfil) es el mejor en 10 líneas y `gbm_v2` en la O.
  - Las conclusiones de un solo periodo de evaluación habrían sido opuestas en DEV y en el holdout: la ventaja del GBM depende de la cantidad de historia.
- **H5, holdout, con el híbrido elegido en DEV sin tocar:**
  - WAPE global 0,0555 frente a 0,0725 del naive.
  - Mejora frente al naive de 11,5 % a 27,6 % en las 11 líneas; en la A, 24,9 % (0,0512 frente a 0,0682).
  - Cobertura del intervalo del 80 % calibrado (banda del GBM): 0,79–0,82. En DEV la calibración quedó conservadora (0,89–0,93), porque el error baja a medida que crece la historia.
- **H6:** el híbrido elegido por WAPE no es mejor para H6. Por ejemplo, en J el AUC es 0,751 frente a 0,910 del GBM. El indicador de día de alta demanda usa el **GBM base pre-registrado** solo en las 7 líneas que cumplieron: A, 2, J, M, O, P y T-A.
  - La exclusión de la Línea 1 por tener menos de 10 positivos es una regla añadida ahora.
- **H5b:** la lluvia sigue descartada. Los días lluviosos favorecen a los modelos con perfil, pero se concentran en abril–mayo de 2024: confusión con el mes, sin efecto atribuible a la lluvia.

## 2026-11-24 — [F10.5–F10.7] Página multilínea, pruebas y cierre de la fase 3
- **Artefactos reconstruidos desde un commit limpio** (`git_commit_build = c1e0481`; el manifiesto ahora marca `+cambios_sin_commit` si el árbol no está limpio).
  - Categorías sobre 8 961 días-línea: anomalías local, multilínea y generalizada = 429 (4,8 %, coherente con el presupuesto del 5 %); no evaluable = 690 (días incompletos de los cables); problema de datos = 113; efecto de medición = 24; ausencia simultánea = 21.
  - Cobertura de la banda que muestra la plataforma (calibrada y centrada en el modelo elegido, 2023-08…2024-06): 0,751–0,831.
- **Página `pages/2_Red_multilinea.py`:**
  - pestaña de red con esquema de relaciones entre líneas (no geográfico), reproducción por hora en el navegador, mapa de calor línea × hora, tabla del motor de reglas y comparación de real frente a esperado;
  - pestaña de pronóstico, con reproducción fuera de muestra y el próximo día (01/07/2024, festivo de San Pedro y San Pablo);
  - pestaña de simulación, etiquetada como tal;
  - pestaña de catálogo, con estados adoptado, descriptivo y descartado, umbrales y lo que no se puede hacer.
  - Cada bloque lleva una etiqueta de dato real, estimación, simulación o predicción.
- **Defectos de UI hallados y corregidos (ver `docs/validacion/F10_fase3_plataforma.md` §4, D4–D7):**
  - contraste de los textos secundarios (axe);
  - la Línea L desaparecía del esquema;
  - un gráfico de 1 051 px quedaba recortado. Lo detectó un verificador E2E endurecido, que ahora también mide el recorte dentro de gráficos y tablas; antes solo medía el desbordamiento de la página;
  - la tabla no cabía en móvil.
- **Pruebas:**
  - 174 rápidas aprobadas, 1 omitida (TensorFlow), cobertura del 83 %;
  - 2 E2E aprobadas: la app original y la página nueva;
  - verificación visual-datos: 44 de 44 celdas idénticas a `dias.csv.gz` en 4 fechas;
  - rendimiento con AppTest: cambio de fecha con p95 de 0,48 s.
- **Incidencia de entorno:** una prueba tardó 606 s por sobresuscripción de hilos mientras corría la construcción de artefactos. En solitario tarda 3 s.
- **Cierre:** `docs/validacion/F10_fase3_plataforma.md` (dictamen) y `docs/fase3/HOJA_DE_RUTA_DATOS.md` (capacidades viables y no viables, y fuentes necesarias en orden de valor).
