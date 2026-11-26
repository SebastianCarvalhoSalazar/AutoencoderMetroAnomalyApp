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

## 2026-11-24 — [F10.6] CI en rojo por importación entre pruebas
- **Fallo:** la ejecución 18 de GitHub Actions falló en la recolección de pruebas (`tests/unit/test_{red,pronostico,plataforma}.py`). Esos archivos importaban `tests.conftest`, lo que funciona con `python -m pytest`, que añade la raíz a `sys.path`, pero no con el ejecutable `pytest` que usa el CI. En local solo se había usado `python -m pytest`.
- **Corrección:** se expone `make_series` como el fixture `series_factory` en `tests/conftest.py`.
- **Verificación:** con `pytest`, igual que en el CI, desde la raíz: 174 aprobadas, 1 omitida, cobertura del 82,8 %.

## 2026-11-24 — [F10.6] CI en verde y PR
- La ejecución 19 de GitHub Actions pasó en sus dos trabajos, `tests` y `research`.
- Se abrió el Draft PR #1 hacia `master`, a pedido del usuario.

## 2026-11-24 — [F10.8] Cierre de pendientes de la fase 3 (a pedido del usuario)
- **Operación diaria multilínea:** `src/metro_anomaly/operacion_red.py` y `run_daily.py --red [--pronostico] [--monitor]`.
  - Evalúa las 11 líneas con la misma lógica causal y los mismos umbrales congelados que los artefactos.
  - Registra la auditoría en JSON Lines de forma idempotente, con clave (fecha, línea, SHA-256 del manifiesto, SHA-256 de los datos).
  - Emite el pronóstico del día siguiente con datos anteriores a la fecha objetivo (reajuste diario).
  - Monitorea, por línea, la tasa de alertas a 90 días (> 10 %), la cobertura del intervalo (fuera de [0,70; 0,90] con ≥ 20 días) y las versiones del manifiesto.
  - Acepta archivos oficiales con varias líneas (`--archivo`). Códigos de salida: 0, 2 y 3.
  - **Verificado:** coincide con la reproducción histórica en 33 de 33 línea-días (`tests/integration/test_operacion_red.py`, 14 pruebas).
- **Pronóstico como biblioteca:** `forecast_day`, `conformal_factor` y `conformal_online` pasan a `pronostico.py`.
  - Al re-ejecutar las métricas del experimento, `reports/fase3/pronostico_*.md` queda idéntico.
  - Para pronósticos nuevos, el factor conformal usa todas las predicciones fuera de muestra de cada línea. Valores: 1,0 en la O y hasta 1,25 en la A; queda en el manifiesto.
- **Hueco de reproducibilidad corregido:** `build_plataforma.py` dependía de un `.pkl` de 24 MB que git ignora, así que un clon limpio no podía reconstruir la plataforma. Ahora usa `reports/fase3/pronostico_oos_holdout.csv.gz` (2,3 MB, versionado).
- **Determinismo:** `scripts/fase3/check_determinism_plataforma.py` hizo dos construcciones en copias aisladas sin `.git`, y las 5 salidas resultaron idénticas entre sí y a las versionadas (`reports/fase3/determinismo_plataforma.md`). Los `.csv.gz` se escriben ahora sin fecha en el encabezado gzip.
- **Procedencia:**
  - Una primera reconstrucción quedó marcada `221e1ea+cambios_sin_commit` porque, durante la construcción, se creó un script nuevo sin versionar. La marca funcionó como se diseñó.
  - Se hizo commit del script y se reconstruyó desde `221e1ea`, ya limpio.
  - Diferencias frente a los artefactos anteriores: categorías y scores idénticos; cambian 44 notas de calendario y el factor conformal del pronóstico del día siguiente.
- **Cierre anticipado del 24 y 31 de diciembre:**
  - Oficial a las 22:00 en 2022 (Telemedellín citando al Metro) y en 2024 (comunicado N.° 2152 del Metro).
  - **2023 solo con prensa** (Pulzo/Q'hubo: 21:00), coherente con los datos (h21 reducida y h22 sin registro).
  - Se registró como OP1–OP3 en `data/eventos_reales.csv` y se anota en `event_note`. Las páginas se revisaron en busca de instrucciones inyectadas y no se hallaron.
  - `reports/externo.md` se regeneró: solo cambia la nota del 24/12/2019 y el ancho de la columna.
- **Animación:** botón "▶ Reproducir el día hora a hora" que dibuja un fotograma por hora (04–22) con la hora en el título. El E2E verifica 5 de 5 fotogramas distintos.
- **Fichas:** `MODEL_CARD.md` (modelos de la fase 3, usos no previstos, operación y mantenimiento) y `DATA_CARD.md` (eventos actualizados, datos multilínea y fuentes externas).
  - Se corrigió una afirmación desactualizada: "no se hallaron incidentes de la Línea A en 2022–2024" ya no era cierta desde F8 (IN1–IN3).

## 2026-11-24 — [F10.8] Simulación de la operación diaria: dos defectos hallados (D9, D10) y un hallazgo operativo
- **Primera simulación de junio de 2024** (`scripts/fase3/simulacion_operacion_red.py`): 309 de 310 línea-días coincidían con la reproducción histórica.
  - **D9:** con una evaluación que empezaba a mitad de mes, `build_history` ajustaba los modelos con datos hasta t−1 y no hasta el inicio del mes, en contra de su propio docstring. El score de la Línea 1 del 08/06/2024 pasó de 5,80 a 4,66 y perdió la alerta.
  - **Corrección:** el ajuste usa siempre datos anteriores al inicio del mes. Los artefactos no cambian, porque siempre se construyen desde el día 1.
  - **Prueba:** `test_mid_month_start_matches_full_month` falla sin la corrección y pasa con ella (verificado).
- **Hallazgo operativo:** la Línea P registró 300–680 entradas diarias del 8 al 13/06/2024, frente a ~22 000 habituales; su WAPE de pronóstico en junio fue de 0,28.
  - **Verificado:** Telemedellín (06/06/2024) anunció el mantenimiento anual de los metrocables: P del 8 al 13, M del 14 al 16 y H del 17 al 19 de junio de 2024.
  - La plataforma había marcado los 6 días de P como anomalía o problema de datos.
- **D10 (falla silenciosa):** las líneas **sin ningún registro** en el día (M del 14 al 16 y H del 17 al 19 de junio de 2024; cierres de 2022) no generaban fila, así que la plataforma no decía nada de una línea cerrada.
  - **Corrección:** se crea una fila cuando la línea suele operar (registro en ≥ 80 % de los 56 días previos; con 90 % de 28 días se dejaba de marcar al tercer día de cierre) y se clasifica "línea sin registro".
  - **Calendario de cierres programados:** `data/cierres_programados.csv` incluye solo los anuncios verificados, de 2022 y 2024; para 2023 solo hay notas sin año explícito. Esos días se clasifican "cierre programado", se excluyen del ajuste de los modelos y de los rezagos del pronóstico, y el pronóstico de un día cerrado es 0.
  - Es información anunciada de antemano (por ejemplo, el 06/06 para el 08/06), así que no hay fuga. Sí exige que alguien cargue los anuncios: es la fuente n.° 3 de la hoja de ruta ("oferta programada").
  - Con la regla nueva aparecen además cierres sin anunciar en el calendario: K del 27 al 30/06/2024, J del 30/10 al 01/11/2023, K del 02 al 04/11/2023, entre otros.

## 2026-11-24 — [F10.8] Cierre: D11, simulación final y verificación completa
- **D11:** el lote fallaba (`KeyError`) al registrar una línea "sin registro", que no tiene fila ese día. Lo halló la simulación del 27/06/2024 (Línea K). Se corrigió con `reindex` y la prueba `test_line_without_records_is_logged`.
- **Cobertura del pronóstico:** los días de cierre programado se pronostican en 0 por regla. Se excluyen de la cobertura y del WAPE del modelo y se informan aparte (en una ejecución intermedia, la Línea P mostraba una cobertura de 0,644 por incluirlos).
- **Simulación final de junio de 2024:**
  - 319/319 línea-días iguales a la reproducción histórica;
  - sin advertencias del monitor;
  - WAPE con reajuste diario equivalente al mensual;
  - WAPE de la Línea P con cierres: 0,092, frente a 0,276 sin el calendario.
- **Artefactos:** reconstruidos (código `16b3ab3`; el manifiesto registra `02d9247`, que solo añade documentación) y con determinismo re-verificado.
- **Pruebas:** 194 rápidas y 2 E2E aprobadas; cobertura del 84,8 %.
- **Informe:** `docs/validacion/F10_fase3_plataforma.md` §7.

## 2026-11-26 — [AE.0] Rama de investigación de autoencoders (a pedido del usuario)
- Se creó la rama `claude/investigacion-autoencoders` desde `6fbdbaf`.
- Se pre-registraron las hipótesis A1–A6 con sus criterios en `docs/investigacion_autoencoders/PLAN.md`, antes de cualquier experimento.
- Punto de partida: `ae_contextual` empata con el perfil contextual en recall moderado, pero duplica la tasa de alertas, y es mejor en errores de datos. El AE original queda por detrás.

## 2026-11-26 — [AE.1] Reproducción del punto de partida
- **Qué se hizo:** reentrenamiento desde cero (rolling mensual, Línea A) de `perfil_contextual/peak_shift`, `ae_contextual/top2`, `ae_contextual/max` y `ae_original/max`, en DEV y en el holdout (`scripts/investigacion_ae/ae1_reproducir.py`).
- **Resultado:** los scores diarios del holdout son **idénticos** a las corridas de la fase 2 (diferencia máxima 0) en las 4 configuraciones.
  - **Defecto en mi primera comprobación:** marcó "no reproduce" en `ae_contextual/max` y `ae_original/max` porque las cifras publicadas de esas dos son promedios de 3 semillas y se compararon con la semilla 0.
  - **Corrección:** con 3 semillas publicadas, el criterio es la igualdad exacta con la semilla 0 guardada. Se recalculó con `--reusar`, sin reentrenar.
- **Hallazgo:** el AE y el perfil son complementarios por tipo de anomalía.
  - El AE es mejor en caídas de 1–3 horas (0,975 frente a 0,745), en aumentos por evento (0,440 frente a 0,253) y en errores de datos (hora duplicada: 0,937 frente a 0,761).
  - El AE es mucho peor en caídas de todo el día (0,151 frente a 0,522) y en un día hábil con perfil de festivo (0,058 frente a 1,000): trabaja sobre el perfil sin nivel y no ve los cambios de nivel.
  - Recall moderado: AE 0,522 frente a perfil 0,506. Diferencia +0,016 [−0,038; 0,069], **sin diferencia demostrada**. Tasa de alertas en días reales: 4,9 % frente a 2,2 %.
- **Implicación para A1:** la combinación tiene una base empírica más amplia que solo los errores de datos. El criterio pre-registrado de A1 **no se cambia**.
- **Visualizaciones:** tablero "Autoencoders en el Metro" (artifact privado), alimentado por `scripts/investigacion_ae/exportar_tablero.py`.

## 2026-11-26 — [AE.1+] Validación y ampliación del tablero de visualizaciones (a pedido del usuario)
- **Validación propia de la página:** la página no reportó ninguna apertura con los datos nuevos, así que se validó en un arnés local.
  - El arnés reproduce el entorno del tablero (`tests/investigacion_ae/arnes_tablero.py`) en un Chromium real, a 1280 y 400 px, en tema claro y oscuro, con los mismos archivos de datos (`tests/investigacion_ae/validar_tablero.js`).
  - **Defectos hallados y corregidos:**
    1. El primer minigráfico de la serie diaria medía su ancho antes de que la cuadrícula ubicara todas las cajas y se dibujaba diminuto.
    2. La última columna de la tabla quedaba cortada.
    3. Los nombres de las tareas mostraban acentos graves del Markdown.
    4. El diagrama de arquitectura se veía ilegible en el móvil y su texto se salía de las cajas.
    5. Unos pocos días extremos aplastaban el espacio latente; se pasó a escala symlog.
    6. Una etiqueta de eje quedaba superpuesta.
    7. Las barras de decisión se medían antes de estar en la página.
  - **Estado final:** sin errores de JavaScript, sin desbordamiento de página y sin dibujos vacíos, en ambos anchos y temas; los 5 ejemplos se recorren sin errores.
- **Nueva sección "Cómo funciona"** (`scripts/investigacion_ae/ae_explicar.py`, con los mismos modelos de la evaluación rolling y la semilla 0):
  - diagrama de la arquitectura (30 entradas → 16 → 8 → **4** → 8 → 16 → 19) y los 5 pasos de la decisión;
  - explorador de 5 días: lo observado frente a lo esperado por cada modelo, |z| por hora y decisión de ambos;
  - distribución del score respecto del umbral;
  - espacio latente: los días se agrupan por tipo, con 91,5 % y 6,9 % de varianza en dos componentes;
  - mapa |z| por día y hora.
  - **Control cruzado:** la proporción de días sobre el umbral en la distribución coincide con AE.1 (moderadas: AE 52,2 % y perfil 50,6 %; días reales: 4,95 % y 2,2 %).
- **Hallazgo:** en el incidente real del 28/08/2023, la caída extrema de las 18–19 h distorsiona toda la reconstrucción del AE.
  - Lo esperado a las 6 h baja a 21 560 frente a 64 208 observados.
  - El AE alerta (score 69,3 frente a un umbral de 9,5), pero señala las 6 h y las 19 h; el perfil señala las 18 h y las 19 h.
  - Un día anómalo contamina la reconstrucción de horas normales, lo que explica la menor precisión del AE al localizar la hora (0,935 frente a 0,983).
- **Defecto hallado por la página real, no por el arnés:** `ae1_metricas.json` contenía `NaN` (la localización del AE original). El JSON estándar no lo admite, así que el tablero rechazó el archivo ("no tiene filas legibles").
  - El arnés no lo vio porque insertaba los datos como JavaScript, donde `NaN` es válido.
  - **Corrección:** el exportador escribe `null` y valida con `allow_nan=False`; el arnés ahora parsea con `JSON.parse`, igual que el tablero.
  - **Verificado:** el arnés detecta el error con el archivo anterior y queda limpio con el nuevo.

## 2026-11-26 — [AE.2] Diseño de la combinación (fijado ANTES de calcular resultados)
- **Regla de combinación:** score combinado = max(score_perfil / umbral_perfil, score_AE / umbral_AE), con los umbrales de DEV de AE.1. El umbral combinado es el p95 de ese máximo en los días reales de DEV (2022-04…2023-06): el mismo presupuesto del 5 %.
  - No se ajusta ningún peso ni reparto: es la forma más simple de unir dos detectores calibrados.
- **Hora señalada:** la del detector cuyo cociente es mayor.
- **Criterio (sin cambios desde `PLAN.md`):**
  1. Recall T4 medio (cero, duplicada, desfase) ≥ +0,05 frente al perfil solo, con IC 95 % > 0.
  2. Recall moderado T1–T3 sin pérdida: el IC del cambio contiene 0 o está por encima.
  - Ambos con diferencias pareadas sobre las mismas inyecciones y bootstrap por semanas.
- **Insumos:** las corridas de AE.1 (semilla 0), sin reentrenar.

## 2026-11-26 — [AE.2] Resultado de A1: NO CUMPLE (descartada)
- **Holdout, combinado − perfil, pareado:**
  - Recall T4 medio: +0,058 [0,027; 0,094]. Cumple su parte.
  - Recall moderado T1–T3: −0,054 [−0,102; −0,006]. El IC está por debajo de 0: hay pérdida y **no cumple**.
- **Causa:** el combinado conserva el presupuesto del 5 % subiendo el umbral efectivo de cada detector (el umbral combinado de DEV es 1,214 veces el de cada uno).
  - El perfil pierde las caídas de todo el día que detectaba cerca del umbral: T2 moderada pasa de 0,522 a 0,165.
  - En el holdout, el combinado alerta en el 4,1 % de los días reales, frente al 2,2 % del perfil, cuyo umbral de DEV resultó conservador.
- **DEV (solo referencia):** T4 +0,040 [0,018; 0,066] y moderado +0,008 [−0,029; 0,048]. En DEV no había pérdida: la conclusión depende del periodo, y decide el holdout pre-registrado.
- **Lo que sí muestra:** en el combinado, el AE aporta la mayoría de las detecciones de T1 y T3, y el perfil la mayoría de las de T2, T4 y T5. La complementariedad es real, pero con una sola cuota de alertas, ganar en un tipo cuesta en otro.
- **No se prueban otras reglas de combinación** sobre el holdout: sería ajustar a la vista del resultado. Otra regla tendría que pre-registrarse y evaluarse con datos nuevos (posteriores a junio de 2024).

## 2026-11-26 — [AE.3] Diseño de A2 (fijado ANTES de entrenar)
- **Modelo `ae_multilinea`:**
  - **Entrada:** el día completo de las 11 líneas principales, es decir, 209 celdas. Cada celda es log1p menos el nivel causal de su línea, estandarizada con la media y el desvío de entrenamiento; una celda sin dato vale 0 (la media). Se añaden los 11 rasgos de calendario de `ae_contextual`.
  - **Red:** 64-32-**8**-32-64 con salida lineal de 209, pérdida MAE, early stopping con validación temporal y semilla 0. Es la escala de `ae_contextual` (16-8-4 para 30 entradas) llevada a 220 entradas.
  - **Score de la línea objetivo:** z = residuo × sd / MAD de los residuos de entrenamiento, en sus 19 columnas; agregación top2, igual que `ae_contextual`.
- **Protocolo:** idéntico al de H1 (`scripts/fase3/exp_red.py`).
  - Holdout rolling 2023-07…2024-06 con ajuste desde 2022-01, e inyecciones solo en la línea objetivo.
  - Umbral de cada detector = p95 de sus scores reales del periodo (mismo presupuesto) y diferencia pareada de recall moderado con IC por bootstrap semanal.
  - Comparación con `coherencia_red`, recalculada en la misma corrida.
- **Criterio (pre-registrado en `PLAN.md`), concretado:**
  1. **Línea A, no inferioridad:** límite inferior del IC de (ae_multilinea − coherencia) ≥ −0,02.
  2. **Mejora en al menos una de B, 1 o T-A:** IC > 0.
  3. **Festivos:** tasa de alertas en festivos de la Línea A ≤ la de la coherencia.

## 2026-11-26 — [AE.4] A3 descartada por premisa falsa (antes de entrenar)
- **Hallazgo en los datos:** en los cables (H, J, K, M, P), las horas 04–07 faltan en el 90–98 % de los domingos y festivos y casi nunca en días hábiles o sábados.
  - El patrón dominante es (4, 5, 6, 7) o (4, 5, 6, 7, 8). La mediana es de 4 horas faltantes por día incompleto.
- **Horario oficial** (metrodemedellin.gov.co/en/travel-with-us/schedule, consultado 2026-11-26; revisado en busca de instrucciones inyectadas, sin hallazgos): los domingos y festivos, la Línea K opera de 8:30 a 22:00 y las H, J, M y P de 9:00 a 22:00.
  - Es el horario vigente; el patrón de los datos de 2022–2024 es coherente con él.
- **Conclusión:** esas horas no son datos perdidos sino **horas sin servicio**. Imputarlas con un AE inventaría demanda que no existe.
  - La hipótesis A3 queda **descartada** sin ejecutarla.
  - Los huecos verdaderos (no estructurales) son muy pocos para evaluar una imputación.
- **Recomendación para la plataforma (fuera de esta rama, sin implementar):** evaluar cada día con el conjunto de horas que la línea opera según su tipo de día, en lugar de exigir las 19. Así se recuperaría la mayoría de los 690 línea-días "no evaluables". No requiere AE.

## 2026-11-26 — [AE.5] y [AE.6] Diseños fijados ANTES de calcular
- **A4 (AE.5, descriptiva):**
  - **Puntuación:** distancia euclidiana de las 4 coordenadas del cuello de cada día a la mediana de los días de DEV de su mismo tipo (controla el tipo de día, que es una entrada del modelo). Se usa el modelo `ae_contextual` del último mes evaluado, ya calculado para el tablero.
  - **Positivos:** los días de la Línea A en 2022-02…2024-06 registrados en `data/eventos_reales.csv`: elecciones EV5–EV7 y EV4/MED1, incidentes IN1–IN3 y cierres anticipados OP1–OP2. No hay cierres programados de la Línea A en `data/cierres_programados.csv`.
  - **Negativos:** el resto de los días completos.
  - **Métrica:** AUC con IC por bootstrap. Como referencia, el mismo AUC con el score del perfil contextual.
  - **Advertencia declarada:** el pre-registro mencionaba "festivos verificados", pero los festivos son una entrada del modelo (calendario), así que separarlos sería trivial y no se usan como positivos.
  - **Criterio:** AUC ≥ 0,85. Solo descriptivo.
- **A5 (AE.6):**
  - **Aviso de cambio de régimen del AE:** la media móvil de 7 días del score diario de `ae_contextual/top2` supera su p99 en días reales de DEV (2022-04…2023-06, corridas de AE.1).
  - **Aviso del monitor actual:** la tasa de alertas de los últimos 90 días del perfil contextual supera el 10 % (`operations.monitor`).
  - **COVID:** corrida causal de 2020 con ajuste desde 2019-01, como en `run_eval.py` periodo `covid`; se compara la fecha del primer aviso de cada uno tras EV1 (2020-03-20).
  - **Falsos avisos:** episodios de aviso del AE en 2022-04…2024-06, contando como episodio cada racha de días consecutivos con aviso.
  - **Criterio (de `PLAN.md`):** el AE avisa ≥ 3 días antes que el monitor y tiene ≤ 1 episodio espurio por año en 2022–2024.

## 2026-11-26 — [AE.3] Resultado de A2: NO CUMPLE (descartada)
- **Recall moderado en el holdout** (mismo presupuesto, inyecciones solo en la línea objetivo), `ae_multilinea` frente a `coherencia_red`:

  | Línea | Coherencia | AE multilínea | Diferencia [IC 95 %] |
  |---|---|---|---|
  | A | 0,900 | 0,759 | −0,141 [−0,182; −0,097] |
  | B | 0,691 | 0,444 | −0,247 [−0,299; −0,195] |
  | 1 | 0,794 | 0,677 | −0,117 [−0,156; −0,073] |
  | T-A | 0,077 | 0,204 | **+0,126 [+0,106; +0,149]** |

- **Criterio:**
  - No inferioridad en A: **no cumple**.
  - Mejora en otra línea: cumple, en T-A.
  - Festivos en A: **no cumple** (0,167 frente a 0,111).
- **Control:** las cifras de la coherencia reproducen exactamente H1 (`reports/fase3/red.md`): A 0,900, B 0,691, 1 0,794 y T-A 0,077.
- **Hallazgo secundario (no adoptable por sí solo):** en el tranvía, donde la coherencia no mejoraba al perfil, el AE multilínea sí detecta más anomalías locales. Valdría pre-registrarlo como hipótesis específica y evaluarlo con datos nuevos; no se adopta a partir de este resultado.

## 2026-11-26 — [AE.5] Resultado de A4 (descriptiva): CUMPLE el criterio puntual, con cautela
- **AUC** de la distancia latente al centro de su tipo de día: **0,877** (exigido ≥ 0,85), con IC 95 % [0,712; 0,985]. Referencia, score del perfil: 0,841 [0,619; 1,0].
- **Cautela:**
  - Solo hay 9 positivos y el IC incluye valores bajo 0,85.
  - El modelo se ajustó con datos hasta mayo de 2024, así que la mayoría de los días son de ajuste y no de evaluación.
  - Los días de 2023-12-24 y 31 se excluyen por no tener las 19 horas completas.
- **Qué separa y qué no:**
  - Bien: las 4 jornadas electorales (percentil 99,7–100) y los cierres anticipados de 2022 (97,5 y 99,4).
  - No: las interrupciones parciales IN1 e IN2 (percentiles 74 y 33).
  - IN3 (28/08/2023) queda en el percentil 83: la reconstrucción falla mucho, pero su resumen latente no es tan atípico.
- **Uso permitido:** descriptivo, para explicar cómo organiza los días el modelo. No se adopta para alertar.
