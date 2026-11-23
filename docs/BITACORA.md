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
