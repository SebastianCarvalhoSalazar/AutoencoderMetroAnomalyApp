# Auditoría técnica end-to-end — AutoencoderMetroAnomalyApp

Fecha: 2026-11-23 · Commit auditado: `3a1948c` (modelo desplegado generado en `98b5725`)
Alcance: `app.py`, `utils.py`, `dev/model.ipynb`, `dev/results/*`, `dev/data_test/*`, datos `data/` y `dev/data/`, `requirements.txt`, README.

**No se modificó ningún archivo original.** Todo lo de esta carpeta (`audit/`) es nuevo: informe, scripts de experimentos y logs.

Leyenda: **[COMPROBADO]** = verificado por ejecución o con evidencia directa en el repositorio · **[HIPÓTESIS]** = plausible, todavía sin confirmar · **[MEJORA]** = oportunidad, no es un error.

---

## 0. Resumen ejecutivo

El proyecto no resuelve hoy el problema que plantea ("detectar anomalías operativas en la afluencia"). Las métricas publicadas (Acc 0,77 / Recall 0,81 / F1 0,78) **no son válidas**, por cinco razones comprobadas:

1. **Las etiquetas de "anomalía" no salen de los residuos de la descomposición.** Un bug en un `merge` hace que el IQR se calcule sobre los **valores crudos** de cada día de la semana. Toda la descomposición aditiva/multiplicativa/STL es código muerto para el etiquetado.
2. **El autoencoder se entrenó con los 912 días**, incluidos los outliers y los días de validación y prueba. El README afirma que se entrenó "solo con datos normales".
3. **El umbral se calibra y se evalúa sobre el mismo conjunto**, con una prevalencia artificial del 52 % (la real es 24,8 %). Sobre el total de días, el modelo desplegado tiene precisión 0,53 y marca como anómalo el 22,9 % de los días normales.
4. **Las métricas reportadas salen de una selección entre muchas ejecuciones sin semilla.** El kernel registra unas 1.150 ejecuciones. Al replicar el protocolo con 32 semillas, la mediana de F1 es 0,755 (rango 0,40–0,83). La combinación publicada está en el ~16 % superior.
5. **Lo que el modelo aprende a imitar es mayormente calendario.** El 95 % de los festivos y el 98 % de los días de diciembre están etiquetados como "anomalía".

Además, la app es vulnerable a **ejecución remota de código** (`pickle.loads` sobre el archivo subido; comprobado con una prueba de concepto inocua). Por otro lado, **173 de los 276 archivos de demo "normales" son restos de ejecuciones anteriores**: con ellos la app acierta solo el 42,9 % de los 389 demos.

Con un protocolo correcto (sin fuga, split temporal y umbral elegido en validación), el autoencoder tal como está diseñado obtiene un AUC de 0,855 ± 0,011. Agregándole el día de la semana como contexto sube a 0,952. Una regla robusta por perfil de día de la semana, sin red neuronal, llega a 0,987. Esta última cifra está inflada porque la regla se parece a la que genera las etiquetas (ver §3.3), y eso confirma que el cuello de botella es la definición de las etiquetas, no la arquitectura.

---

## 1. Qué hace el sistema realmente (reconstrucción verificada)

| Etapa | Lo que dice el README / el comentario | Lo que hace el código (verificado) |
|---|---|---|
| Datos | Afluencia del Metro | Solo Línea A, 2022-01-01 → 2024-06-30 (912 días, 20 horas: 04h–23h). Se descartan 2019–2021 y las otras 10 líneas |
| Imputación | Rellenar faltantes | Solo faltan valores en 23h (310/912 = 34 %) y en 22h (2). Se rellenan con la mediana por día de la semana de todo el período, futuro incluido |
| Descomposición | Elegir la mejor entre aditiva, multiplicativa y STL | Siempre gana la multiplicativa, porque se comparan MSE con unidades distintas. Además su resultado no se usa para etiquetar (ver C1) |
| Etiquetas | IQR sobre residuos por día de la semana | IQR 1,5 sobre **valores crudos** por día de la semana; 23h nunca se evalúa; día anómalo = al menos 1 hora fuera de rango |
| Entrenamiento | AE con datos normales | AE 20-16-8-4-8-16-20 entrenado con **los 912 días**; scaler ajustado con los 912 |
| Umbral | ROC | `mean + 0.5·std` del error de los normales de validación, evaluado en esa misma validación; no se calcula ninguna ROC |
| App | Carga datos, preprocesa y clasifica | Recibe un vector ya escalado en `.pkl`, sin validarlo; umbral 827 escrito a mano en el código |

Reproducción: `audit/experiments/e1_labels.py` regenera las etiquetas **idénticas al 100 %** a `dev/results/full_2019_2024_data_linea_A_outliers.xlsx` (686 normales / 226 outliers, valores y etiquetas iguales celda a celda).

---

## 2. Hallazgos priorizados

### CRÍTICOS

**C1. Las etiquetas se calculan sobre valores crudos, no sobre residuos.** [COMPROBADO]
- Código: en `detectar_outliers_en_residuos` (notebook, celda 3), `pd.DataFrame(descomposicion.resid.dropna()).merge(data, on='Día')` produce las columnas `['Día','resid', ..., '04:00:00', ...]`. Luego `residuo_estacional = residuo[...][hora]` toma la columna `hora` de `data`, que contiene los **valores originales**. El residuo queda en la columna `resid`, que nunca se usa.
- Evidencia (`e12_resid_bug.py`): la reimplementación con valores crudos coincide al **100 %** con las etiquetas del repositorio. Con los residuos reales, la coincidencia baja a 82,1 %: cambian 162 de 906 días.
- Consecuencia: la descomposición (y la selección entre aditiva, multiplicativa y STL) no influye en las etiquetas; solo descarta 3 días en cada extremo. La etiqueta mide en realidad "¿este día está lejos del nivel típico de su día de la semana en 2,5 años?". Por eso diciembre queda marcado el 98 % de las veces (con residuos reales sería 56 %) y enero el 53 %.

**C2. Fuga total de datos: el modelo se entrenó con todos los días, incluidos outliers, validación y prueba.** [COMPROBADO]
- Código (celda 17): `train_normales_scaled = scaler.fit_transform(outliers_residuos_agrupados_aligned_all[horas])`. El nombre dice "normales", pero el contenido son las 912 filas. `train_normales` (480 filas) nunca se usa para entrenar.
- Evidencia: la salida guardada del notebook dice `Train size (normal only, scaled): (912, 20)`. El scaler desplegado tiene `n_samples_seen_ = 912`, y `data_min_` y `data_max_` son exactamente el mínimo y el máximo de los 912 días (`e0_deployed.py`).
- Consecuencia: todas las métricas son in-sample, y la premisa del método ("el AE solo conoce lo normal") se rompe, porque el AE también aprendió a reconstruir los outliers.

**C3. Protocolo de evaluación inválido: el mismo conjunto sirve para calibrar el umbral y para medir, con prevalencia artificial.** [COMPROBADO]
- El umbral (`mean+0.5σ` de los normales de `val`) y las métricas se calculan sobre `val`. El conjunto llamado `test` (solo normales) se usa para el *early stopping*. No existe ningún conjunto independiente.
- `val` tiene 113 outliers y 103 normales (52 % de outliers); la prevalencia real es 24,8 %.
- Evidencia (modelo desplegado, umbral 827, sobre los 912 días): TN=529, FP=157, FN=49, TP=177 → **precisión 0,53**, recall 0,78, **FPR 22,9 %**. El 0,77 de precisión publicado se debe a la prevalencia inflada.

**C4. Sesgo de selección y falta de reproducibilidad.** [COMPROBADO]
- No hay semillas (`train_test_split` sin `random_state`, ni semilla para TF). En el commit `98b5725` el kernel registra unas 1.150 ejecuciones (`execution_count` entre 1153 y 1185) y el modelo se llama `functional_31`. En `dev/results/` hay 4 modelos con nombres según sus métricas (69_28_30, 72_58_52, 76_42_48, 77_81_78).
- Experimento (`e6_seeds_original.py`, 32 semillas, protocolo original idéntico):

| métrica | mediana | p5 | p95 | mín–máx | valor publicado | P(≥ publicado) |
|---|---|---|---|---|---|---|
| Accuracy | 0,748 | 0,595 | 0,803 | 0,546–0,815 | 0,77 | 0,34 |
| Recall | 0,743 | 0,461 | 0,830 | 0,283–0,867 | 0,81 | 0,16 |
| F1 | 0,755 | 0,543 | 0,817 | 0,395–0,828 | 0,78 | 0,31 |
| Umbral (pasajeros) | 916 | 765 | 8.675 | 724–8.871 | 827 | – |

  P(las tres a la vez ≥ publicado) = 0,16. El modelo publicado es un extremo favorable de una distribución muy amplia, no un resultado típico. En algunas ejecuciones el umbral llega a 8.871, señal de entrenamientos degenerados.
- El notebook en HEAD mezcla salidas de **dos kernels distintos**: las celdas 0–14 se re-ejecutaron (contadores 1 y 14–25) después de cambiar la descomposición en `3a1948c`, y las celdas 15–33 conservan salidas del kernel anterior (contadores 1166–1185). Ejecutado de principio a fin, el notebook no regenera los artefactos del repositorio.

**C5. Las etiquetas son heurísticas, y lo que el modelo imita es sobre todo calendario predecible.** [COMPROBADO]
- 45 festivos en el período (fuente: librería `holidays`, calendario de Colombia): el **93–95 % está etiquetado como anomalía**, y el 18,6 % de los días anómalos son festivos. Diciembre: 98 %.
- Una regla que solo mira el calendario ("¿es festivo o diciembre?") obtiene AUC 0,71 (split aleatorio) y 0,76 (split temporal) frente a estas etiquetas (`e3_proper.py`).
- Consecuencia conceptual: el AUC del AE mide cuánto se parece su salida a una regla IQR, no su capacidad de detectar incidentes operativos. No hay ninguna verdad de terreno independiente (fallas, cierres, eventos).

**C6. Los archivos de demo están corruptos o desactualizados y la app clasifica mal.** [COMPROBADO]
- `dev/data_test/` tiene 276 normales y 113 outliers. Solo los normales 1–103 y los outliers 1–113 coinciden **exactamente** con filas del dataset usando el scaler desplegado. Los normales 104–276 (173 archivos) quedaron de ejecuciones anteriores con otro rango de datos y otro escalador; el notebook nunca borra los archivos viejos. Su error de emparejamiento mediano es de 16.789 pasajeros.
- Con el modelo desplegado: en los 389 archivos, **accuracy 0,429** (201 de 276 normales marcados como "Anormal"); en los 216 válidos, 0,773.
- Prueba end-to-end con Streamlit y Playwright (`e10_app.js`): `x_test_normal_1.pkl` → "Anormal" (MAE 1049), `x_test_normal_150.pkl` → "Anormal" (MAE 5556). Captura: `audit/app_normal_1_marcado_anormal.png`.
- Los archivos de demo forman parte del conjunto de entrenamiento (consecuencia de C2).

**C7. Ejecución remota de código en la app.** [COMPROBADO]
- `utils.leer_dato` hace `pickle.loads(uploaded_file.getvalue())` sobre un archivo subido por el usuario. Una prueba de concepto inocua (`e11_edge.py`, un `__reduce__` que ejecuta `echo` y crea un archivo) **se ejecutó**. Cualquiera que pueda subir un archivo a la app desplegada puede ejecutar comandos en el servidor.

### ALTOS

**A1. La hora 23:00 nunca se evalúa y además está imputada en un 34 %.** [COMPROBADO] / interpretación [HIPÓTESIS]
- `if hora != list(descomposiciones.keys())[-1]` excluye la última hora (23h): su tasa de outliers es 0,0 %, frente a ~11 % en las demás.
- Los NaN de 23h se concentran en domingos (88 de 131) y festivos (38 de 45). [HIPÓTESIS] Probablemente significan "sin servicio o sin registros" (un cero estructural) y no un faltante aleatorio; habría que confirmarlo con el Metro. Imputarlos con la mediana del día de la semana inventa pasajeros: por ejemplo, 45 en un lunes festivo.
- Los 2 NaN de 22h (24/12 y 31/12) son, con alta probabilidad, días de cierre anticipado, y se imputan como si fueran normales.

**A2. La selección del método de descomposición compara magnitudes incompatibles.** [COMPROBADO]
- El MSE del residuo multiplicativo vale ≈1 (residuos centrados en 1: `mean²+var`), mientras que los de la aditiva y la STL están en pasajeros² (10⁵–10⁷). Ejemplo a las 07h: aditiva 4,84e7, multiplicativa 1,04, STL 2,97e7. **La multiplicativa gana siempre** y la STL añadida en el último commit no puede ganar nunca. (Por C1, además, no tiene efecto.)

**A3. Umbral absoluto en pasajeros + MinMax fijo = sin generalización temporal.** [COMPROBADO]
- `e7_oos.py`: el modelo desplegado aplicado a años no vistos, con etiquetas generadas por la misma heurística:
  - 2019: marca el **96 %** de los días (FPR 0,96), AUC 0,774.
  - 2021: marca el **82 %** (FPR 0,79), AUC 0,681.
- La afluencia media diaria cambia entre años (2019: 635.597; 2022: 568.281; 2023: 595.817). Cualquier cambio de nivel desplaza el MAE y deja inservible el umbral 827. El sistema no tiene recalibración ni monitoreo de deriva.

**A4. La métrica de puntuación no es coherente con la etiqueta.** [COMPROBADO]
- El MAE se calcula en pasajeros: las horas 17h y 18h aportan cada una ~10 % del error, mientras que 04h aporta 2,5 % y 23h 0,4 %. La etiqueta, en cambio, trata todas las horas por igual (IQR por hora). Una anomalía en una hora de bajo flujo pasa casi invisible para el score.
- Normalizar el error por hora mejora el AUC en +0,015/+0,016. Ayuda poco; el problema principal es otro (A5).

**A5. El AE no recibe el contexto del que depende la etiqueta.** [COMPROBADO]
- La etiqueta está condicionada al día de la semana, pero el AE solo ve las 20 horas. Puede reconstruir perfectamente un perfil de domingo aunque el día sea un martes.
- Inyección sintética (`e4_inject.py`, días hábiles normales de prueba): el escenario "perfil de domingo en día hábil" lo detecta el **0 %** de las veces tanto el modelo desplegado como los AE y PCA sin contexto. El AE condicionado al día de la semana lo detecta el **100 %**.
- Agregar el día de la semana como entrada: AUC de 0,855 a 0,952 (temporal) y de 0,883 a 0,949 (aleatorio).

**A6. Una red neuronal no supera a líneas base simples.** [COMPROBADO] (con salvedad)
- Protocolo correcto (`e3_proper.py`: scaler y modelos ajustados solo con días normales de entrenamiento; umbral elegido por F1 en validación; métricas en un test independiente; 5 semillas):

| Método | AUC temporal | PR-AUC temporal | F1 temporal | AUC aleatorio |
|---|---|---|---|---|
| Perfil robusto por día de la semana (max\|z\|, log1p, mediana/MAD) | **0,987** | 0,969 | 0,902 | **0,984 ± 0,006** |
| IsolationForest (+ día de la semana) | 0,956 ± 0,007 | 0,908 | 0,772 | 0,926 |
| AE + día de la semana, error normalizado | 0,952 ± 0,004 | 0,918 | 0,789 | 0,949 |
| AE con error normalizado por hora | 0,871 ± 0,011 | 0,810 | 0,710 | 0,898 |
| **AE original (MAE en pasajeros)** | **0,855 ± 0,011** | 0,783 | 0,673 | 0,883 |
| PCA k=4 | 0,835 | 0,720 | 0,618 | 0,865 |
| Solo calendario (festivo o diciembre) | 0,761 | 0,640 | 0,686 | 0,709 |

  Split temporal: entrenamiento con normales de 2022; validación en 2023-S1; prueba de 2023-07 a 2024-06 (prevalencia 24,6 %).
- Salvedad: la línea base de perfil robusto se parece a la regla que genera las etiquetas, así que su AUC está inflado por construcción. Esa es justamente la conclusión: **con estas etiquetas, el problema es casi determinista a partir de (horas, día de la semana)**. La red no aporta valor, y la validez del sistema depende de redefinir qué es una anomalía.

**A7. El umbral `mean + 0.5σ` es arbitrario e implica muchos falsos positivos por diseño.** [COMPROBADO]
- Con k=0,5, una fracción importante de los normales queda por encima del umbral: FPR de 22,9 % sobre todos los días y de 24 % sobre los normales de prueba en el experimento de inyección. Para un sistema de alertas eso equivale a ~1 falsa alarma cada 4 días. No hay justificación ni curva costo/beneficio.

**A8. Dependencias no instalables y entorno no declarado.** [COMPROBADO]
- `pip install -r requirements.txt` falla en Linux: `tensorflow-intel==2.17.0` solo tiene wheels para `win_amd64`.
- Incluye `torch`, `torchaudio` y `torchvision` (varios GB) que no se usan. Faltan `statsmodels` y `openpyxl`, que el notebook necesita. La celda 2 del notebook muestra un `pip install` fallido por espacios en la ruta de Windows.
- Para auditar se usó Python 3.10.5 con las versiones fijadas, sin `tensorflow-intel` ni `torch`, más las dependencias faltantes (`audit/requirements-audit.txt`).

### MEDIOS

**M1. La app no valida entradas y no coincide con lo que dice el README.** [COMPROBADO] (`e11_edge.py`)
- Un vector con NaN produce "Anormal" en silencio (`nan < 827` es `False`). Longitud 19 o 21, forma (1,20), lista o texto terminan en una traza de TensorFlow/NumPy expuesta al usuario.
- Recibe vectores **ya escalados**: el usuario no puede subir datos reales de afluencia. El README dice que la app "limpia y normaliza", y no lo hace. Si se suben conteos crudos, el MAE es de 3,2·10⁹ y la respuesta es "Anormal".
- El README menciona gráficas de la curva de entrenamiento y de la proporción de outliers en la app; no existen.

**M2. Ineficiencias de la app.** [COMPROBADO]
- El modelo y el scaler se cargan en cada interacción, sin `st.cache_resource`: latencia medida de ~3,4 s por archivo.
- `predict` se ejecuta dos veces (`app.py:41` y dentro de `predecir`).
- `UMBRAL=827` está escrito a mano en lugar de leerse de `tf_modelo_77_81_78_threshold.yml`, con riesgo de desalinearse si se cambia el modelo.
- `keras.models.load_model` emite una advertencia de optimizador (el `compile_config` guardado no incluye el optimizador). No afecta la inferencia; se elimina con `compile=False`.

**M3. Las etiquetas no son causales.** [COMPROBADO]
- `seasonal_decompose` usa una media móvil centrada (±3 días) y los cuantiles e imputaciones se calculan con todo el período. El enfoque no es compatible con el trabajo futuro de detección en tiempo real que describe el README.

**M4. Los datos disponibles están subutilizados.** [COMPROBADO] (los datos) / [MEJORA]
- Están disponibles 2019 (prepandemia) y 11 líneas por año. Se usa solo la Línea A de 2022–2024S1. Con varias líneas se podría distinguir un evento de todo el sistema (festivo, paro) de una falla propia de una línea.
- Verificado: `Total general` = suma de las horas en el 100 % de las filas de 2023, así que eliminar esa columna no pierde información.

**M5. Ruido de etiqueta por la regla "alguna hora".** [COMPROBADO]
- La tasa por hora (~11 %) es muy superior al 0,7 % que se esperaría con datos gaussianos y 1,5·IQR. 46 días son "anómalos" por **una sola** hora fuera de rango. La distribución de `Total_Outliers` es bimodal: muchos días con 1–2 horas y otros con 17–19.

**M6. Ventanas deslizantes inválidas** (celdas 27–28, hoy sin uso). [COMPROBADO]
- `crear_dataset_con_ventanas` se aplica a `val`, que está barajado (`sample(frac=1)`): concatena días no consecutivos.

**M7. Los días de los extremos quedan etiquetados como "normales" por defecto.** [COMPROBADO]
- Los 3 primeros y los 3 últimos días no tienen residuo; el `fillna(0)` los etiqueta como normales sin evaluarlos.

**M8. Muestra pequeña y sin intervalos de confianza.** [COMPROBADO]
- 216 días en validación; la desviación estándar de F1 entre semillas es 0,095. Las métricas se reportan sin incertidumbre.

### BAJOS
- B1. Restos de una plantilla: "corresponde a un **sujeto**", eje "Índice" en lugar de "Hora".
- B2. `data/` y `dev/data/` son idénticos (mismos md5, 3,3 MB duplicados).
- B3. Código muerto o engañoso: `roc_curve` y `auc` se importan y no se usan (el README habla de ROC); `test_scaled = scaler.transform(train_normales)` se sobrescribe en la línea siguiente.
- B4. `fillna(method='ffill')` está deprecado (FutureWarning en la propia salida del notebook).
- B5. Criterio inconsistente en el empate: la app usa `perdida < umbral` → normal; el notebook usa `errors > threshold` → anómalo.
- B6. El archivo de 2023 contiene una fila del 2024-01-01 (Línea B) que también está en el de 2024. No afecta a la Línea A, pero duplicaría datos si se usan otras líneas.

---

## 3. Cuestionamiento del enfoque (fundamentos)

1. **¿Un AE es la herramienta adecuada?** Un vector de 20 horas por día, ~900 muestras y una estructura dominada por el día de la semana y el calendario es un problema de baja dimensión y fuerte estacionalidad. Un modelo de "perfil esperado" condicionado (día de la semana, festivo, mes y tendencia) da un residuo interpretable por hora y, con estas etiquetas, rinde mejor (A6). Un AE sin contexto, en cambio, no puede representar "normal **para este día**" (A5).
2. **¿Qué es una anomalía?** Hoy se mezclan efectos de calendario conocidos (festivos, temporada decembrina, vacaciones de enero) con incidentes reales. Para la operación, lo útil es "desviación respecto de lo esperado **dado el calendario**". Los festivos deberían ser una covariable, no una anomalía.
3. **Supuestos estadísticos.** El IQR por día de la semana sobre valores crudos de 2,5 años supone estacionariedad del nivel, que no se cumple (hay tendencia y recuperación pospandemia). El 1,5·IQR aplicado a 19 horas sin corrección por comparaciones múltiples infla las etiquetas positivas.
4. **Coherencia entre objetivos, método y código.** El objetivo (detectar anomalías operativas en tiempo real) choca con un etiquetado no causal (M3), una evaluación in-sample (C2, C3) y una app que no recibe datos reales (M1).

---

## 4. Plan de acción propuesto (nada implementado; requiere aprobación)

### Fase 0 — Seguridad y demo (inmediata, bajo esfuerzo)
1. Reemplazar `pickle` por CSV/JSON con esquema: 20 valores finitos ≥ 0, en pasajeros crudos y con fecha. *Justificación: C7.*
2. Escalar dentro de la app; validar forma y NaN con mensajes claros; usar `st.cache_resource`; un solo `predict`; leer el umbral del `.yml`; `compile=False`. *Justificación: M1, M2.*
3. Regenerar `dev/data_test/` desde cero (borrando los archivos obsoletos) con días de un **test no visto**. *Justificación: C6.*
4. `requirements.txt` multiplataforma: quitar `tensorflow-intel` y `torch*`; agregar `statsmodels` y `openpyxl`. *Justificación: A8.*

### Fase 1 — Redefinir y corregir las etiquetas
5. Corregir C1 (usar `resid`), incluir 23h (A1), decidir la semántica de los NaN de 23h con el Metro y quitar el `fillna(0)` de los días de los extremos (M7).
6. Separar el calendario: festivos (y eventualmente temporadas) como covariables o como una categoría aparte, no como anomalía (C5).
7. Idealmente, construir una **verdad de terreno independiente** (registros de incidentes, cierres y eventos) aunque sea pequeña, para validar. Sin ella, cualquier métrica mide el parecido a una heurística.

### Fase 2 — Protocolo de validación
8. Split temporal (train < val < test), con scaler y modelo ajustados solo con los normales de entrenamiento, umbral elegido en validación según un presupuesto de alertas (FPR objetivo) y test intocado. *Justificación: C2, C3.*
9. Semillas fijas; reportar media ± desviación o IC sobre ≥10 semillas, con AUC, PR-AUC y métricas a la prevalencia real; prohibir elegir el modelo según métricas de test. *Justificación: C4, M8.*
10. Pasar el notebook a scripts o módulos (`src/`) ejecutables de principio a fin, con salidas versionadas. *Justificación: C4, reproducibilidad.*

### Fase 3 — Modelado (con evidencia de §2-A6)
11. Línea base obligatoria: perfil robusto por día de la semana/festivo en log1p, con z-scores por hora (ya rinde AUC ≈ 0,98).
12. Si se mantiene el AE: condicionarlo al calendario (día de la semana, festivo, mes), normalizar el error por hora y comparar siempre contra la línea base.
13. Para el objetivo en tiempo real: un pronóstico de la hora siguiente con regresores de calendario (por ejemplo, regresión robusta o gradient boosting) y alertas sobre residuos estandarizados. Eso es causal por diseño. Opcionalmente, multilínea para separar eventos sistémicos de los locales (M4).

### Fase 4 — Operación
14. Umbrales relativos o recalibración periódica, monitoreo de deriva de nivel (A3) y registro de alertas con retroalimentación del operador.

---

## 5. Reproducibilidad de esta auditoría
- Entorno: Python 3.10.5 + `audit/requirements-audit.txt` (TF 2.17.0, Keras 3.5.0, sklearn 1.5.2, pandas 2.2.3, statsmodels 0.14.4, holidays 0.58).
- Scripts en `audit/experiments/`; salidas en `audit/logs/`. Las rutas apuntan a `/home/user/AutoencoderMetroAnomalyApp` (ajustar `REPO` en `pipeline.py`, `common.py` y los scripts para otro entorno).
  - `e1_labels.py`: reproducción exacta de las etiquetas y diagnóstico de datos.
  - `e12_resid_bug.py`: demostración de C1.
  - `e0_deployed.py`: modelo y scaler desplegados, archivos de demo, aporte de cada hora al MAE.
  - `e6_seeds_original.py`: protocolo original con 32 semillas (C4).
  - `e3_proper.py`: protocolo correcto, splits aleatorio y temporal, comparación de métodos.
  - `e4_inject.py`: anomalías sintéticas.
  - `e7_oos.py`: generalización a 2019 y 2021.
  - `e11_edge.py`: casos límite y prueba de concepto de pickle (la PoC escribe `/tmp/pwned_poc.txt`; no hace nada más).
  - `e10_app.js`: prueba end-to-end de la app con Playwright.
- Limitaciones: (i) no existe verdad de terreno independiente, así que todas las métricas se refieren a la heurística; (ii) en la inyección sintética, el modelo desplegado se evalúa sobre días que vio en entrenamiento (las versiones inyectadas sí son nuevas); (iii) 5 semillas en el protocolo correcto (las desviaciones son pequeñas, ≤0,02 en AUC); (iv) los festivos provienen de la librería `holidays`, no de un decreto oficial; (v) la semántica de los NaN de 23h no se pudo confirmar con una fuente oficial.
