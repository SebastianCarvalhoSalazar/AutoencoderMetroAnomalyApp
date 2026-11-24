# Informe final de validación

Fecha: 2026-11-23 · Rama: `claude/repository-technical-audit-89x93t` · Plan: [`docs/PLAN_MAESTRO.md`](../PLAN_MAESTRO.md) · Historia: [`docs/BITACORA.md`](../BITACORA.md)

## 1. Veredicto
El sistema es **funcional de extremo a extremo**: datos validados, modelo causal, evaluación reproducible y app operativa. Además es **claramente superior al sistema original**, y su utilidad se demostró con una evaluación metodológicamente válida: sin fuga, con separación temporal, umbral congelado, líneas base, eventos reales y casos límite.

**No se declara éxito pleno del modelo:** incumple tres criterios pre-registrados (D2 severas, D2 moderadas y D3). Los criterios no se relajaron.

## 2. Criterios de terminado (pre-registrados en el commit `30e9939`, antes de implementar)

### Modelo
| ID | Meta | Resultado | Estado | Evidencia |
|---|---|---|---|---|
| D1 | alertas 2–10 % | 2,2 % [0,8–3,6] | ✔ | `reports/holdout.md` |
| D2 | severas ≥ 0,95 · moderadas ≥ 0,80 | 0,894 · 0,506 | **✘ · ✘** | `reports/holdout.md` |
| D3 | festivos ≤ 10 % | 11,1 % (2/18) | **✘** | `reports/holdout.md` |
| D4 | T5 ≥ 0,95 | 1,00 | ✔ | `reports/holdout.md` |
| D5 | no peor que la mejor línea base; mejor que el AE original | no inferior; +0,207 [IC > 0] | ✔ | `reports/holdout.md` |
| D6 | 2019 ≤ 15 % | 7,5 % | ✔ | `reports/externo.md` |
| D7 | eventos reales ≥ 2/3 | 4/4 formal; **reinterpretado en F8**: las elecciones son un efecto de medición (no cobro); frente a incidentes operacionales verificados, 1/3 | ✔ formal / ✘ sustantivo | `reports/externo.md`, `F8_F9_produccion_y_ux.md` |
| D8 | causalidad | tests en verde para los 8 detectores (6 sin TF + 2 AE) | ✔ | `tests/unit/test_detectors*.py` |
| D9 | determinismo | dos ejecuciones aisladas: 8/8 salidas con SHA-256 idéntico | ✔ | `reports/determinismo.md` |

### Pipeline de datos (P1–P4) ✔
Esquema validado en los 6 archivos; duplicado resuelto; reporte de calidad; semántica de las 22/23 h decidida (horario dominical oficial no verificable); cobertura de `data.py` 96 %.

### Aplicación (A1–A5) ✔
Sin pickle de usuario; fecha + xlsx/csv; validación con mensajes; 0 trazas; AppTest y E2E en verde; p95 de 665 ms.

### Reproducibilidad e infraestructura (R1–R5)
- R1 ✔: instalación limpia con `uv` y con `pip` en GitHub Actions.
- R2 ✔: `make reproduce`; una copia limpia reproduce exactamente la selección, el umbral y las métricas de las 15 configuraciones sin AE.
- R3 ✔: el artefacto incluye SHA-256 de los datos, commit, umbral y métricas.
- R4 ✔: CI en verde en las ejecuciones verificadas.
- R5 ✔: cobertura del 85 % con un mínimo exigido del 80 %.

### Documentación ✔
README, ficha del modelo, ficha de los datos, informes F0–F5 y este informe; app y notebook originales marcados como legacy.

## 3. Lo que cambió respecto del sistema auditado
| Aspecto | Antes | Ahora |
|---|---|---|
| Etiquetas | IQR sobre valores crudos (bug), usadas como verdad | Ya no son verdad de terreno (κ entre variantes 0,48–0,54; 95 % de festivos marcados); se valida con inyección y eventos reales |
| Fuga | Modelo y scaler entrenados con los 912 días | Rolling mensual causal; test de no-fuga y de causalidad |
| Umbral | `mean+0,5σ` evaluado en el mismo conjunto; 827 pasajeros fijos | Presupuesto de alertas en DEV; congelado; en z (escala relativa) |
| Contexto | Ninguno (AE ciego al calendario) | Tipo de día, festivos, Semana Santa, temporada, nivel causal |
| Generalización | 96 % de alertas en 2019 | 7,5 % en 2019 |
| Seguridad | RCE vía pickle | Sin pickle; artefacto JSON; entradas validadas |
| Reproducibilidad | Notebook mezclado entre kernels, sin semillas, dependencias de Windows | Scripts, semillas, dependencias fijadas, CI, determinismo verificado |

## 4. Determinismo (D9)
**DETERMINISTA ✔.** Dos ejecuciones completas de `scripts/reproduce.py` en copias aisladas dan SHA-256 idénticos en las 8 salidas: selección, informes DEV/holdout/externo, calidad, etiquetas y artefacto (sin campos volátiles). Además, la copia limpia reproduce exactamente la selección, el umbral (7,3880478521949176) y las métricas de holdout de las 15 configuraciones sin AE que están versionadas.

## 5. Desviaciones del plan (todas registradas en la bitácora)
1. **Ronda 2 de selección en DEV** después de un resultado negativo en la ronda 1 (declarada antes de ejecutarla; única).
2. **Defecto de implementación descubierto después de abrir el holdout**: nivel NaN a principios de enero en los detectores contextuales, y métricas que contaban los NaN como "sin alerta". Se corrigió, se repitió la selección con la misma regla y se hizo una **segunda lectura del holdout**. Las dos versiones están publicadas (`reports/v1_con_defecto/`). La mejora aparente de la ronda 2 estaba inflada por el defecto.
3. **F2.5 / P3 parcial**: sin acceso a fuentes oficiales del horario del Metro ni al texto de la Ley 51/1983 (DNS bloqueado en el entorno). Se compensó con un contraste empírico.
4. **F4.5 (multilínea)** no se ejecutó: su condición previa (cumplir F4.3) no se dio.
5. Se añadieron eventos electorales a D7 después de ver DEV. Solo cuentan los **prospectivos** (2019 y 2023, registrados antes de ver esos periodos).

## 5b. Fase 2 (preparación para producción y UX)
Ver `docs/validacion/F8_F9_produccion_y_ux.md` y `docs/validacion/DICTAMEN_PRODUCCION.md`: **no apto como detección temprana de incidentes; apto con condiciones como apoyo retrospectivo y de calidad de datos.**

## 6. Pendientes explícitos y trabajo futuro (no evaluado)
- Mejorar la sensibilidad a aumentos por evento (T3) y a desplazamientos leves del día; candidatos: recalibración periódica del umbral, agregación por bloques (CUSUM), contexto multilínea.
- Tipo de día propio para Navidad y Año Nuevo (D3).
- Manejo de cambios de régimen (readaptación rápida o modo "régimen excepcional").
- Construir una verdad de terreno operativa con registros de incidentes del Metro y retroalimentación de los operadores; es la mayor limitación de validez externa.
- Verificar el calendario y los horarios contra fuentes oficiales cuando haya acceso.
- Confirmar las causas de las alertas reales sin fuente (por ejemplo, 2023-08-28 de 17 a 19 h, 2023-12-14).
