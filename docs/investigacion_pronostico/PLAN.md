# Investigación: corrección por persistencia del error en el pronóstico (hipótesis P1)

Estado: **pre-registrado** el 2026-11-27, antes de calcular cualquier resultado de P1. Diseño, datos y criterio quedan fijos en este
documento; cualquier cambio posterior se registra en `docs/BITACORA.md` como desviación.

## Motivación (y su debilidad)

Diagnóstico exploratorio (no pre-registrado) sobre los residuos del perfil contextual, Línea A, holdout 2023-07…2024-06 con
reentrenamiento mensual causal (364 días con las 19 horas): la desviación media diaria (media de z) tiene autocorrelación por
rangos (Spearman) de 0,623 en el rezago 1 y 0,475 en el rezago 7, aunque la de Pearson es 0,114 (dominada por días extremos:
máx. 53,3 frente a p99 5,8). Es decir: si el modelo se queda corto un día, tiende a quedarse corto los siguientes. El pronóstico
actual (GBM / híbrido) usa el último día comparable, pero **no** el error de ayer.

Debilidad declarada: la hipótesis se sugirió mirando el periodo holdout (aunque con otro modelo, el perfil, y una sola línea).
Por eso el criterio incluye además una réplica en un periodo anterior (DEV tardío, 2023-01…2023-06) que no se miró.

## Hipótesis P1

Corregir el pronóstico vigente con un término AR(1) sobre su error medio del día anterior reduce el WAPE del día siguiente.

## Datos

- Predicciones fuera de muestra ya generadas por `scripts/fase3/exp_pronostico.py` (origen rodante mensual; ajuste con
  2022-01…M−1; rasgos causales a resolución diaria): `reports/fase3/pronostico_predicciones_dev.pkl` (2022-07…2023-06) y
  `reports/fase3/pronostico_predicciones_holdout.pkl` (2023-07…2024-06). 11 líneas (sin la L).
- **Comprobación previa (si falla, se detiene):** el WAPE del pronóstico vigente recalculado desde esos archivos debe coincidir
  con `reports/fase3/pronostico_holdout.json` (`seleccion_dev`, global y por línea, a 4 decimales).

## Diseño (fijo)

1. Pronóstico base b(t, h): el modelo elegido en DEV para cada línea (`seleccion_por_linea` de `pronostico_holdout.json`).
2. Error e(t, h) = log1p(y) − log1p(b). Error del día ē(t) = media de e(t, h) en las horas con dato (mínimo 10 de 19; si no, sin dato).
3. Predictor x(t) = clip(ē(t − 1), −0,3, 0,3), con t − 1 = el día calendario anterior. x(t) = 0 (sin corrección) si ē(t − 1) no
   existe o si t − 1 fue un cierre programado de la línea (`data/cierres_programados.csv`, conocido de antemano).
   El dato de t − 1 está disponible al emitir el pronóstico de t (cierre del día t − 1).
4. Coeficiente φ por línea y mes M: regresión de Huber sin intercepto (`HuberRegressor(fit_intercept=False)`, parámetros por
   defecto) de ē(t) sobre x(t) con todos los días fuera de muestra de los meses < M desde 2022-07; φ recortado a [0, 1];
   φ = 0 si hay menos de 60 días.
5. Pronóstico corregido: ŷ(t, h) = expm1(log1p(b(t, h)) + φ · x(t)), el mismo ajuste en todas las horas del día.
6. Filas evaluadas: las mismas del informe vigente (filas del holdout con predicción de todos los modelos; 69 252).

## Criterio de éxito (todos)

Sea r = 1 − WAPE(corregido) / WAPE(base).

1. Holdout 2023-07…2024-06, las 11 líneas juntas: r ≥ 2 %.
2. IC 95 % de r por bootstrap de semanas ISO completas (1000 réplicas, semilla 0; se remuestrean semanas con las filas de todas
   las líneas): límite inferior > 0.
3. Ninguna línea empeora más de 1 % relativo (WAPE corregido ≤ 1,01 × WAPE base en cada línea).
4. Réplica en DEV tardío 2023-01…2023-06 (φ estimado con los meses fuera de muestra anteriores de DEV): r > 0.

Si cumple: se propone (no se aplica sin decisión) incorporarlo a `forecast_day`. Si no cumple: se registra y se descarta.

## Descriptivo (no entra en el veredicto)

r por línea; WAPE en días posteriores a festivos; AUC de días de alta demanda (H6) con el pronóstico corregido; φ por línea y mes.
La banda del 80 % no se modifica en esta prueba.

## Resultado (2026-11-27, calculado después de fijar lo anterior)

**P1: NO CUMPLE (descartada).** Detalle en `reports/investigacion_pronostico/p1.md`. Reproducción previa del WAPE vigente: correcta
(0,0555; 69 252 filas).

| Criterio | Exigido | Obtenido | ¿Cumple? |
|---|---|---|---|
| 1. r en holdout | ≥ 2 % | +0,40 % | no |
| 2. IC 95 % de r | límite inferior > 0 | [−2,37 %; +3,72 %] | no |
| 3. Ninguna línea peor > 1 % | todas | 5 de 11 empeoran más de 1 % (H −4,7 %, K −4,7 %, P −3,9 %, O −2,0 %, J −1,8 %) | no |
| 4. Réplica DEV 2023-01…2023-06 | r > 0 | −3,51 % [−6,61 %; −0,23 %] | no |

Los φ estimados son positivos (0,10–0,41): el error sí persiste dentro de la muestra de ajuste, pero trasladarlo al día siguiente no
reduce el error fuera de muestra. Descriptivo, sin valor confirmatorio: el AUC de días de alta demanda (H6) sube en 9 de 11 líneas
y el WAPE de los días posteriores a festivo baja de 0,0798 a 0,0775. El pronóstico vigente no se modifica.
