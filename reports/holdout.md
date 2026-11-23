# Evaluación en HOLDOUT (generado por `scripts/evaluate_holdout.py`)

Periodo: 2023-07-01…2024-06-30 · configuración elegida en DEV: **perfil_contextual/peak_shift** · umbral congelado: 7.3880

## Criterios pre-registrados (configuración elegida)

| Criterio | Valor | Cumple |
|---|---|---|
| D1 tasa de alertas ∈ [2 %, 10 %] | 0.022 | ✔ |
| D2 recall severas ≥ 0,95 | 0.894 | ✘ |
| D2 recall moderadas ≥ 0,80 | 0.506 | ✘ |
| D3 alertas en festivos ≤ 10 % | 0.111 | ✘ |
| D4 recall T5 ≥ 0,95 | 1.000 | ✔ |
| D5a no peor que la mejor línea base (IC sup. de la dif. ≥ 0) | -0.016 | ✔ |
| D5b supera al AE original (IC inf. de la dif. > 0) | 0.207 | ✔ |

IC 95 % (bootstrap por semanas, 2000 réplicas):

|                          |   valor |   ic_inf |   ic_sup |
|:-------------------------|--------:|---------:|---------:|
| tasa_alertas             |   0.022 |    0.008 |    0.036 |
| tasa_alertas_festivos    |   0.111 |    0     |    0.294 |
| recall_severa            |   0.894 |    0.867 |    0.92  |
| recall_moderada          |   0.506 |    0.462 |    0.549 |
| recall_T5_perfil_festivo |   1     |    1     |    1     |
| recall_moderada_media    |   0.506 |    0.462 |    0.549 |

## Todas las configuraciones (cada una con su umbral de DEV)

| config                          |   umbral_dev |   n_semillas |   tasa_alertas |   tasa_alertas_festivos |   recall_severa |   recall_moderada |   recall_leve |   recall_moderada_media |   recall_T4_cero |   recall_T4_duplicada |   recall_T4_desfase |   recall_T5_perfil_festivo |   localizacion_hora |   auc_vs_resid_k1.5 |   auc_vs_raw_k1.5 |
|:--------------------------------|-------------:|-------------:|---------------:|------------------------:|----------------:|------------------:|--------------:|------------------------:|-----------------:|----------------------:|--------------------:|---------------------------:|--------------------:|--------------------:|------------------:|
| ae_contextual/top2              |        9.529 |            1 |          0.049 |                   0.111 |           0.879 |             0.522 |         0.2   |                   0.522 |            1     |                 0.937 |               1     |                      0.058 |               0.935 |               0.766 |             0.88  |
| perfil_contextual/peak_shift    |        7.388 |            1 |          0.022 |                   0.111 |           0.894 |             0.506 |         0.124 |                   0.506 |            1     |                 0.761 |               0.948 |                      1     |               0.983 |               0.711 |             0.804 |
| ae_contextual/max               |       11.294 |            3 |          0.043 |                   0.13  |           0.867 |             0.485 |         0.185 |                   0.485 |            1     |                 0.995 |               1     |                      0.079 |               0.932 |               0.756 |             0.898 |
| regresion_contextual/peak_shift |        8.641 |            1 |          0.014 |                   0     |           0.898 |             0.442 |         0.159 |                   0.442 |            1     |                 0.772 |               0.97  |                      1     |               0.99  |               0.713 |             0.765 |
| perfil_contextual/top2          |        8.501 |            1 |          0.025 |                   0.111 |           0.857 |             0.429 |         0.131 |                   0.429 |            1     |                 0.522 |               0.948 |                      1     |               0.98  |               0.719 |             0.824 |
| regresion_contextual/max        |        9.119 |            1 |          0.011 |                   0     |           0.876 |             0.399 |         0.091 |                   0.399 |            1     |                 0.75  |               0.956 |                      1     |               0.991 |               0.714 |             0.78  |
| perfil_contextual/max           |       10.027 |            1 |          0.019 |                   0.111 |           0.871 |             0.372 |         0.119 |                   0.372 |            1     |                 0.753 |               0.945 |                      0.946 |               0.983 |               0.717 |             0.819 |
| perfil_robusto/peak_shift       |        7.889 |            1 |          0.055 |                   0.111 |           0.885 |             0.375 |         0.1   |                   0.375 |            1     |                 0.734 |               1     |                      1     |               0.951 |               0.736 |             0.829 |
| regresion_contextual/top2       |        8.399 |            1 |          0.014 |                   0     |           0.817 |             0.368 |         0.087 |                   0.368 |            1     |                 0.431 |               0.907 |                      1     |               0.988 |               0.718 |             0.78  |
| ae_original/max                 |     3578.13  |            3 |          0.024 |                   0     |           0.539 |             0.304 |         0.072 |                   0.304 |            0.571 |                 0.496 |               0.809 |                      0     |             nan     |               0.705 |             0.823 |
| perfil_robusto/max              |        9.987 |            1 |          0.058 |                   0.111 |           0.889 |             0.367 |         0.12  |                   0.367 |            1     |                 0.78  |               0.997 |                      1     |               0.95  |               0.764 |             0.893 |
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
| ae_contextual/top2              |       -0.016 |   -0.069 |    0.038 | 1092 |
| ae_contextual/max               |        0.012 |   -0.042 |    0.066 | 1092 |
| regresion_contextual/peak_shift |        0.064 |    0.033 |    0.096 | 1092 |
| perfil_contextual/top2          |        0.077 |    0.045 |    0.109 | 1092 |
| regresion_contextual/max        |        0.107 |    0.076 |    0.138 | 1092 |
| perfil_contextual/max           |        0.135 |    0.105 |    0.163 | 1092 |
| perfil_robusto/peak_shift       |        0.131 |    0.075 |    0.183 | 1092 |
| regresion_contextual/top2       |        0.138 |    0.103 |    0.173 | 1092 |
| ae_original/max                 |        0.207 |    0.16  |    0.251 | 1092 |
| perfil_robusto/max              |        0.139 |    0.077 |    0.194 | 1092 |
| naive_semanal/max               |        0.218 |    0.166 |    0.268 | 1092 |
| perfil_robusto/top2             |        0.193 |    0.127 |    0.254 | 1092 |
| naive_semanal/top2              |        0.245 |    0.188 |    0.298 | 1092 |
| regresion_calendario/peak_shift |        0.168 |    0.119 |    0.213 | 1092 |
| regresion_calendario/max        |        0.186 |    0.138 |    0.23  | 1092 |
| regresion_calendario/top2       |        0.271 |    0.22  |    0.319 | 1092 |
| isolation_forest/max            |        0.342 |    0.274 |    0.404 | 1092 |

## Alertas del modelo elegido en días reales del holdout (8)

| date                |   score | tipo_dia        | festivo   | horas_principales (z)                           |   z_medio | dia      |
|:--------------------|--------:|:----------------|:----------|:------------------------------------------------|----------:|:---------|
| 2023-08-28 00:00:00 |   65.02 | habil           |           | 19h(-66.4), 18h(-45.2), 17h(-12.6), 22h(-3.0)   |     -7.33 | Monday   |
| 2023-09-23 00:00:00 |   13.01 | sabado          |           | 12h(-15.0), 05h(-12.6), 04h(-11.6), 06h(-10.4)  |     -5.17 | Saturday |
| 2023-10-29 00:00:00 |  101.01 | domingo_festivo |           | 06h(-110.9), 16h(-98.0), 15h(-89.1), 07h(-86.2) |    -53.33 | Sunday   |
| 2023-12-09 00:00:00 |    9.28 | sabado          |           | 05h(-12.4), 06h(-10.1), 07h(-9.0), 04h(-8.3)    |     -1.77 | Saturday |
| 2023-12-14 00:00:00 |   23.62 | habil           |           | 08h(+22.1), 04h(+22.0), 06h(+21.7), 09h(+20.5)  |     15.85 | Thursday |
| 2023-12-25 00:00:00 |    8.67 | domingo_festivo | Navidad   | 06h(-11.8), 07h(-10.7), 04h(-10.1), 05h(-8.8)   |     -5.19 | Monday   |
| 2024-01-01 00:00:00 |   10.68 | domingo_festivo | Año Nuevo | 06h(-13.3), 07h(-12.4), 04h(-11.2), 08h(-10.1)  |     -6.71 | Monday   |
| 2024-01-11 00:00:00 |    7.92 | habil           |           | 06h(-9.6), 08h(-9.1), 09h(-7.1), 05h(-6.8)      |     -5.39 | Thursday |
