# AE.1 — Reproducción del punto de partida (generado por `scripts/investigacion_ae/ae1_reproducir.py`)

Línea A, rolling mensual; umbral = p95 de los scores reales de DEV de cada configuración; métricas en el holdout 2023-07…2024-06.

|                              |   umbral_dev |   tasa_alertas |   tasa_alertas_festivos |   recall_severa |   recall_moderada |   recall_leve |   recall_T1_moderada |   recall_T2_moderada |   recall_T3_moderada |   recall_T4_cero |   recall_T4_duplicada |   recall_T4_desfase |   recall_T5_perfil_festivo |   localizacion_hora |   semillas_publicadas |   max_dif_score_vs_cache | reproduce_publicado   |
|:-----------------------------|-------------:|---------------:|------------------------:|----------------:|------------------:|--------------:|---------------------:|---------------------:|---------------------:|-----------------:|----------------------:|--------------------:|---------------------------:|--------------------:|----------------------:|-------------------------:|:----------------------|
| perfil_contextual/peak_shift |       7.388  |          0.022 |                   0.111 |           0.894 |             0.506 |         0.124 |                0.745 |                0.522 |                0.253 |            1     |                 0.761 |               0.948 |                      1     |               0.983 |                     1 |                        0 | True                  |
| ae_contextual/top2           |       9.5285 |          0.049 |                   0.111 |           0.879 |             0.522 |         0.2   |                0.975 |                0.151 |                0.44  |            1     |                 0.937 |               1     |                      0.058 |               0.935 |                     1 |                        0 | True                  |
| ae_contextual/max            |      11.2944 |          0.049 |                   0.111 |           0.875 |             0.495 |         0.195 |                0.997 |                0.137 |                0.349 |            1     |                 0.997 |               1     |                      0.058 |               0.94  |                     3 |                        0 | True                  |
| ae_original/max              |    3578.13   |          0.036 |                   0     |           0.562 |             0.299 |         0.082 |                0.42  |                0.036 |                0.442 |            0.571 |                 0.459 |               0.832 |                      0     |             nan     |                     3 |                        0 | True                  |

## Diferencia pareada de recall moderado frente al perfil contextual

- ae_contextual/top2: {'diferencia': 0.016, 'ic_inf': -0.038, 'ic_sup': 0.069, 'n': 1092}
- ae_contextual/max: {'diferencia': -0.012, 'ic_inf': -0.066, 'ic_sup': 0.042, 'n': 1092}
- ae_original/max: {'diferencia': -0.207, 'ic_inf': -0.251, 'ic_sup': -0.16, 'n': 1092}

Tiempos de ejecución (s): {'dev:perfil_contextual/peak_shift': 13.4, 'dev:ae_contextual/top2': 228.7, 'dev:ae_contextual/max': 209.0, 'dev:ae_original/max': 303.5, 'holdout:perfil_contextual/peak_shift': 12.2, 'holdout:ae_contextual/top2': 234.9, 'holdout:ae_contextual/max': 245.8, 'holdout:ae_original/max': 267.8}