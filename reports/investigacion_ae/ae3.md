# AE.3 — Hipótesis A2: autoencoder multilínea frente a coherencia de red (generado por `scripts/investigacion_ae/ae3_multilinea.py`)

Holdout 2023-07…2024-06, inyecciones solo en la línea objetivo, mismo presupuesto (p95 de los scores reales del periodo).

## LINEA A

|                          |   coherencia |   ae_multilinea |
|:-------------------------|-------------:|----------------:|
| tasa_alertas             |       0.0522 |          0.0522 |
| tasa_alertas_festivos    |       0.1111 |          0.1667 |
| recall_moderada          |       0.9002 |          0.7592 |
| recall_severa            |       0.9856 |          0.9984 |
| recall_leve              |       0.5815 |          0.283  |
| localizacion_hora        |       0.9819 |          0.9781 |
| recall_T1_moderada       |       0.956  |          0.9973 |
| recall_T2_moderada       |       0.9148 |          0.3736 |
| recall_T3_moderada       |       0.8297 |          0.9066 |
| recall_T4_cero           |       1      |          1      |
| recall_T4_duplicada      |       0.9368 |          0.989  |
| recall_T4_desfase        |       0.9863 |          1      |
| recall_T5_perfil_festivo |       1      |          1      |

Diferencia pareada de recall moderado (ae_multilinea − coherencia): {'diferencia': -0.141, 'ic_inf': -0.1824, 'ic_sup': -0.0967, 'n': 1092}

## LINEA B

|                          |   coherencia |   ae_multilinea |
|:-------------------------|-------------:|----------------:|
| tasa_alertas             |       0.0525 |          0.0525 |
| tasa_alertas_festivos    |       0.1111 |          0.1667 |
| recall_moderada          |       0.6906 |          0.4438 |
| recall_severa            |       0.9407 |          0.949  |
| recall_leve              |       0.1971 |          0.1556 |
| localizacion_hora        |       0.9669 |          0.9539 |
| recall_T1_moderada       |       0.8315 |          0.8895 |
| recall_T2_moderada       |       0.7707 |          0.1077 |
| recall_T3_moderada       |       0.4696 |          0.3343 |
| recall_T4_cero           |       1      |          1      |
| recall_T4_duplicada      |       0.8481 |          0.6657 |
| recall_T4_desfase        |       0.9475 |          1      |
| recall_T5_perfil_festivo |       1      |          1      |

Diferencia pareada de recall moderado (ae_multilinea − coherencia): {'diferencia': -0.2468, 'ic_inf': -0.2988, 'ic_sup': -0.1947, 'n': 1086}

## LINEA 1

|                          |   coherencia |   ae_multilinea |
|:-------------------------|-------------:|----------------:|
| tasa_alertas             |       0.0522 |          0.0522 |
| tasa_alertas_festivos    |       0.1111 |          0.1111 |
| recall_moderada          |       0.794  |          0.6767 |
| recall_severa            |       0.9707 |          0.993  |
| recall_leve              |       0.369  |          0.2491 |
| localizacion_hora        |       0.9752 |          0.9748 |
| recall_T1_moderada       |       0.9093 |          0.9835 |
| recall_T2_moderada       |       0.8242 |          0.2445 |
| recall_T3_moderada       |       0.6484 |          0.8022 |
| recall_T4_cero           |       1      |          1      |
| recall_T4_duplicada      |       0.8984 |          0.9533 |
| recall_T4_desfase        |       0.9835 |          1      |
| recall_T5_perfil_festivo |       1      |          1      |

Diferencia pareada de recall moderado (ae_multilinea − coherencia): {'diferencia': -0.1172, 'ic_inf': -0.1563, 'ic_sup': -0.0729, 'n': 1092}

## LINEA T-A

|                          |   coherencia |   ae_multilinea |
|:-------------------------|-------------:|----------------:|
| tasa_alertas             |       0.0525 |          0.0525 |
| tasa_alertas_festivos    |       0.1667 |          0.1667 |
| recall_moderada          |       0.0773 |          0.2035 |
| recall_severa            |       0.5769 |          0.6142 |
| recall_leve              |       0.0552 |          0.0681 |
| localizacion_hora        |       0.8787 |          0.8818 |
| recall_T1_moderada       |       0.1077 |          0.4586 |
| recall_T2_moderada       |       0.0746 |          0.0856 |
| recall_T3_moderada       |       0.0497 |          0.0663 |
| recall_T4_cero           |       1      |          1      |
| recall_T4_duplicada      |       0.0773 |          0.1077 |
| recall_T4_desfase        |       0.4199 |          0.6271 |
| recall_T5_perfil_festivo |       0.9461 |          1      |

Diferencia pareada de recall moderado (ae_multilinea − coherencia): {'diferencia': 0.1262, 'ic_inf': 0.1061, 'ic_sup': 0.1486, 'n': 1086}

## Criterio pre-registrado

- Línea A no inferior (IC inf ≥ −0,02): {'cumple': False, 'ic_inf': -0.1824, 'exigido': '≥ −0,02'}
- Mejora en B, 1 o T-A (IC > 0): {'cumple': True, 'lineas': ['LINEA T-A'], 'exigido': 'IC > 0 en B, 1 o T-A'}
- Festivos A no peor: {'cumple': False, 'ae': 0.1667, 'coherencia': 0.1111}
- **Veredicto: NO CUMPLE**