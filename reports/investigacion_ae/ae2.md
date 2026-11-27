# AE.2 — Hipótesis A1: perfil + autoencoder (generado por `scripts/investigacion_ae/ae2_combinacion.py`)

Regla: max(s_perfil/u_perfil, s_AE/u_AE) > p95 de DEV; umbral combinado (DEV) = 1.2143.

|                   |   tasa_alertas |   tasa_alertas_festivos |   recall_moderada |   recall_severa |   recall_leve |   localizacion_hora |   recall_T1_moderada |   recall_T2_moderada |   recall_T3_moderada |   recall_T4_cero |   recall_T4_duplicada |   recall_T4_desfase |   recall_T1_severa |   recall_T2_severa |   recall_T3_severa |   recall_T5_perfil_festivo |   recall_T4_medio |
|:------------------|---------------:|------------------------:|------------------:|----------------:|--------------:|--------------------:|---------------------:|---------------------:|---------------------:|-----------------:|----------------------:|--------------------:|-------------------:|-------------------:|-------------------:|---------------------------:|------------------:|
| dev|perfil        |         0.0504 |                  0.0833 |            0.432  |          0.8859 |        0.1645 |              0.9635 |               0.7654 |               0.2785 |               0.2522 |                1 |                0.7149 |              0.9671 |             0.9825 |             0.8618 |             0.7259 |                     0.9804 |            0.894  |
| dev|ae            |         0.0504 |                  0.2083 |            0.4832 |          0.8018 |        0.1886 |              0.921  |               0.9167 |               0.0658 |               0.4671 |                1 |                0.8377 |              1      |             1      |             0.5548 |             0.9561 |                     0      |            0.9459 |
| dev|combinado     |         0.0504 |                  0.0833 |            0.4401 |          0.9313 |        0.1477 |              0.9413 |               0.9211 |               0.1096 |               0.2895 |                1 |                0.8004 |              1      |             1      |             0.8794 |             0.875  |                     0.9804 |            0.9335 |
| holdout|perfil    |         0.022  |                  0.1111 |            0.5064 |          0.8945 |        0.1236 |              0.9831 |               0.7445 |               0.522  |               0.2527 |                1 |                0.761  |              0.9478 |             0.9863 |             0.9203 |             0.6813 |                     1      |            0.9029 |
| holdout|ae        |         0.0495 |                  0.1111 |            0.522  |          0.8788 |        0.1996 |              0.9354 |               0.9753 |               0.1511 |               0.4396 |                1 |                0.9368 |              1      |             1      |             0.8819 |             1      |                     0.0579 |            0.9789 |
| holdout|combinado |         0.0412 |                  0.1111 |            0.4524 |          0.9534 |        0.1456 |              0.9488 |               0.956  |               0.1648 |               0.2363 |                1 |                0.8819 |              1      |             1      |             0.9121 |             0.9313 |                     0.9463 |            0.9606 |

## Criterio pre-registrado (holdout, combinado − perfil)

- Recall T4 medio: {'diferencia': 0.0577, 'ic_inf': 0.0273, 'ic_sup': 0.0938, 'n': 1092} → cumple (exigido ≥ +0,05 con IC > 0)
- Recall moderado T1–T3: {'diferencia': -0.054, 'ic_inf': -0.1019, 'ic_sup': -0.0056, 'n': 1092} → no cumple (exigido IC ∋ 0 o > 0)
- **Veredicto: NO CUMPLE**

Referencia en DEV (no decide): T4 {'diferencia': 0.0395, 'ic_inf': 0.0182, 'ic_sup': 0.0664, 'n': 1368}; moderado {'diferencia': 0.008, 'ic_inf': -0.0292, 'ic_sup': 0.0483, 'n': 1368}

Origen de las alertas reales del combinado en el holdout: {'autoencoder': 10, 'perfil': 5}

Origen de las detecciones por tipo: {'T1': {'autoencoder': 473, 'perfil': 357}, 'T2': {'autoencoder': 84, 'perfil': 328}, 'T3': {'autoencoder': 313, 'perfil': 133}, 'T4': {'autoencoder': 373, 'perfil': 676}, 'T5': {'autoencoder': 2, 'perfil': 227}}