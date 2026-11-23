# Evaluación en HOLDOUT (generado por `scripts/evaluate_holdout.py`)

Periodo: 2023-07-01…2024-06-30 · configuración elegida en DEV: **perfil_contextual/top2** · umbral congelado: 7.3482

## Criterios pre-registrados (configuración elegida)

| Criterio | Valor | Cumple |
|---|---|---|
| D1 tasa de alertas ∈ [2 %, 10 %] | 0.025 | ✔ |
| D2 recall severas ≥ 0,95 | 0.877 | ✘ |
| D2 recall moderadas ≥ 0,80 | 0.603 | ✘ |
| D3 alertas en festivos ≤ 10 % | 0.111 | ✘ |
| D4 recall T5 ≥ 0,95 | 0.955 | ✔ |
| D5a no peor que la mejor línea base (IC sup. de la dif. ≥ 0) | 0.065 | ✔ |
| D5b supera al AE original (IC inf. de la dif. > 0) | 0.303 | ✔ |

IC 95 % (bootstrap por semanas, 2000 réplicas):

|                          |   valor |   ic_inf |   ic_sup |
|:-------------------------|--------:|---------:|---------:|
| tasa_alertas             |   0.025 |    0.011 |    0.041 |
| tasa_alertas_festivos    |   0.111 |    0     |    0.294 |
| recall_severa            |   0.877 |    0.814 |    0.927 |
| recall_moderada          |   0.603 |    0.549 |    0.651 |
| recall_T5_perfil_festivo |   0.955 |    0.893 |    0.996 |
| recall_moderada_media    |   0.603 |    0.549 |    0.651 |

## Todas las configuraciones (cada una con su umbral de DEV)

| config                          |   umbral_dev |   n_semillas |   tasa_alertas |   tasa_alertas_festivos |   recall_severa |   recall_moderada |   recall_leve |   recall_moderada_media |   recall_T4_cero |   recall_T4_duplicada |   recall_T4_desfase |   recall_T5_perfil_festivo |   localizacion_hora |   auc_vs_resid_k1.5 |   auc_vs_raw_k1.5 |
|:--------------------------------|-------------:|-------------:|---------------:|------------------------:|----------------:|------------------:|--------------:|------------------------:|-----------------:|----------------------:|--------------------:|---------------------------:|--------------------:|--------------------:|------------------:|
| perfil_contextual/top2          |        7.348 |            1 |          0.025 |                   0.111 |           0.877 |             0.603 |         0.19  |                   0.603 |            0.951 |                 0.67  |               0.912 |                      0.955 |               0.982 |               0.745 |             0.85  |
| perfil_contextual/peak_shift    |        7.471 |            1 |          0.019 |                   0.111 |           0.865 |             0.538 |         0.134 |                   0.538 |            0.951 |                 0.75  |               0.909 |                      0.955 |               0.983 |               0.733 |             0.821 |
| ae_contextual/top2              |        9.529 |            1 |          0.049 |                   0.111 |           0.879 |             0.522 |         0.2   |                   0.522 |            1     |                 0.937 |               1     |                      0.058 |               0.935 |               0.766 |             0.88  |
| perfil_contextual/max           |        9.162 |            1 |          0.025 |                   0.167 |           0.867 |             0.462 |         0.17  |                   0.462 |            0.951 |                 0.78  |               0.912 |                      0.926 |               0.981 |               0.745 |             0.846 |
| ae_contextual/max               |       11.294 |            3 |          0.043 |                   0.13  |           0.867 |             0.485 |         0.185 |                   0.485 |            1     |                 0.995 |               1     |                      0.079 |               0.932 |               0.756 |             0.898 |
| perfil_robusto/peak_shift       |        7.889 |            1 |          0.055 |                   0.111 |           0.885 |             0.375 |         0.1   |                   0.375 |            1     |                 0.734 |               1     |                      1     |               0.951 |               0.736 |             0.829 |
| regresion_contextual/peak_shift |       10.354 |            1 |          0.014 |                   0     |           0.833 |             0.369 |         0.067 |                   0.369 |            0.951 |                 0.712 |               0.901 |                      0.955 |               0.989 |               0.728 |             0.754 |
| regresion_contextual/max        |       11.199 |            1 |          0.008 |                   0     |           0.8   |             0.322 |         0.031 |                   0.322 |            0.951 |                 0.679 |               0.813 |                      0.955 |               0.993 |               0.743 |             0.78  |
| ae_original/max                 |     3578.13  |            3 |          0.024 |                   0     |           0.539 |             0.304 |         0.072 |                   0.304 |            0.571 |                 0.496 |               0.809 |                      0     |             nan     |               0.705 |             0.823 |
| perfil_robusto/max              |        9.987 |            1 |          0.058 |                   0.111 |           0.889 |             0.367 |         0.12  |                   0.367 |            1     |                 0.78  |               0.997 |                      1     |               0.95  |               0.764 |             0.893 |
| regresion_contextual/top2       |        9.752 |            1 |          0.011 |                   0     |           0.723 |             0.282 |         0.042 |                   0.282 |            0.951 |                 0.217 |               0.766 |                      0.955 |               0.989 |               0.747 |             0.787 |
| naive_semanal/max               |       11.999 |            1 |          0.047 |                   0.222 |           0.859 |             0.288 |         0.06  |                   0.288 |            1     |                 0.659 |               0.997 |                      1     |               0.95  |               0.714 |             0.768 |
| perfil_robusto/top2             |        9.207 |            1 |          0.058 |                   0.111 |           0.816 |             0.313 |         0.103 |                   0.313 |            1     |                 0.313 |               0.997 |                      1     |               0.943 |               0.761 |             0.894 |
| naive_semanal/top2              |       10.285 |            1 |          0.055 |                   0.222 |           0.808 |             0.262 |         0.079 |                   0.262 |            1     |                 0.239 |               0.997 |                      1     |               0.937 |               0.719 |             0.771 |
| regresion_calendario/peak_shift |        8.485 |            1 |          0.036 |                   0.167 |           0.957 |             0.339 |         0.043 |                   0.339 |            1     |                 0.887 |               0.995 |                      1     |               0.976 |               0.745 |             0.814 |
| regresion_calendario/max        |        9.827 |            1 |          0.036 |                   0.167 |           0.917 |             0.321 |         0.041 |                   0.321 |            1     |                 0.86  |               0.995 |                      1     |               0.975 |               0.765 |             0.87  |
| regresion_calendario/top2       |        9.433 |            1 |          0.03  |                   0.167 |           0.788 |             0.235 |         0.034 |                   0.235 |            1     |                 0.071 |               0.915 |                      1     |               0.972 |               0.766 |             0.87  |
| isolation_forest/max            |        0.551 |            1 |          0.038 |                   0.278 |           0.319 |             0.164 |         0.078 |                   0.164 |            0.071 |                 0.049 |               0.78  |                      0.058 |             nan     |               0.715 |             0.82  |

## Diferencia pareada de recall moderado (elegido − otro)

|                                 |   diferencia |   ic_inf |   ic_sup |    n |
|:--------------------------------|-------------:|---------:|---------:|-----:|
| perfil_contextual/peak_shift    |        0.065 |    0.038 |    0.093 | 1092 |
| ae_contextual/top2              |        0.081 |    0.016 |    0.141 | 1092 |
| perfil_contextual/max           |        0.141 |    0.116 |    0.167 | 1092 |
| ae_contextual/max               |        0.108 |    0.044 |    0.165 | 1092 |
| perfil_robusto/peak_shift       |        0.227 |    0.154 |    0.292 | 1092 |
| regresion_contextual/peak_shift |        0.234 |    0.205 |    0.261 | 1092 |
| regresion_contextual/max        |        0.28  |    0.25  |    0.309 | 1092 |
| ae_original/max                 |        0.303 |    0.251 |    0.35  | 1092 |
| perfil_robusto/max              |        0.235 |    0.162 |    0.3   | 1092 |
| regresion_contextual/top2       |        0.321 |    0.289 |    0.35  | 1092 |
| naive_semanal/max               |        0.314 |    0.255 |    0.369 | 1092 |
| perfil_robusto/top2             |        0.289 |    0.212 |    0.36  | 1092 |
| naive_semanal/top2              |        0.341 |    0.279 |    0.395 | 1092 |
| regresion_calendario/peak_shift |        0.264 |    0.205 |    0.315 | 1092 |
| regresion_calendario/max        |        0.282 |    0.226 |    0.332 | 1092 |
| regresion_calendario/top2       |        0.367 |    0.308 |    0.419 | 1092 |
| isolation_forest/max            |        0.439 |    0.363 |    0.507 | 1092 |

## Alertas del modelo elegido en días reales del holdout (9)

| date                |   score | tipo_dia        | festivo   | horas_|z|>umbral                                                                                                                                                                                                                 | dia      |
|:--------------------|--------:|:----------------|:----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
| 2023-08-05 00:00:00 |    8.23 | sabado          |           | 10h(+8.1), 11h(+7.5), 12h(+8.4), 13h(+7.9)                                                                                                                                                                                       | Saturday |
| 2023-08-28 00:00:00 |   57.88 | habil           |           | 17h(-12.8), 18h(-47.6), 19h(-68.2)                                                                                                                                                                                               | Monday   |
| 2023-09-23 00:00:00 |   14.94 | sabado          |           | 04h(-12.8), 05h(-13.3), 06h(-10.6), 07h(-11.2), 12h(-16.6), 13h(-9.1)                                                                                                                                                            | Saturday |
| 2023-10-29 00:00:00 |  107.1  | domingo_festivo |           | 04h(-71.2), 05h(-80.4), 06h(-114.6), 07h(-92.7), 08h(-62.3), 09h(-59.3), 10h(-64.6), 11h(-61.3), 12h(-70.8), 13h(-65.8), 14h(-80.4), 15h(-82.2), 16h(-99.6)                                                                      | Sunday   |
| 2023-12-09 00:00:00 |   11.59 | sabado          |           | 04h(-8.6), 05h(-12.5), 06h(-10.7), 07h(-9.9)                                                                                                                                                                                     | Saturday |
| 2023-12-14 00:00:00 |   22.96 | habil           |           | 04h(+23.5), 05h(+11.6), 06h(+22.4), 07h(+14.9), 08h(+22.3), 09h(+20.8), 10h(+19.1), 11h(+17.8), 12h(+16.4), 13h(+14.6), 14h(+12.5), 15h(+11.8), 16h(+18.8), 17h(+21.2), 18h(+15.5), 19h(+13.5), 20h(+13.7), 21h(+8.6), 22h(+9.2) | Thursday |
| 2023-12-25 00:00:00 |   12.49 | domingo_festivo | Navidad   | 04h(-12.0), 05h(-9.7), 06h(-12.6), 07h(-12.4), 08h(-7.6)                                                                                                                                                                         | Monday   |
| 2024-01-01 00:00:00 |   13.77 | domingo_festivo | Año Nuevo | 04h(-12.7), 05h(-11.6), 06h(-14.0), 07h(-13.5), 08h(-9.7), 09h(-8.3), 10h(-8.6)                                                                                                                                                  | Monday   |
| 2024-03-09 00:00:00 |    8.98 | sabado          |           | 13h(-8.1), 14h(-9.9)                                                                                                                                                                                                             | Saturday |
