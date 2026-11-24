# Experimentos de red: H1–H4 (generado por `scripts/fase3/exp_red.py`)

## H1 — Detector de coherencia entre líneas vs. perfil contextual (mismo presupuesto: 5 % de días reales)

### LINEA A|dev

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.05  |            0.05  |
| tasa_alertas_festivos |               0.083 |            0.042 |
| recall_moderada       |               0.432 |            0.685 |
| recall_severa         |               0.886 |            0.943 |
| recall_leve           |               0.164 |            0.175 |
| recall_moderada_media |               0.432 |            0.685 |
| localizacion_hora     |               0.964 |            0.969 |
| recall_T1_moderada    |               0.765 |            0.825 |
| recall_T2_moderada    |               0.279 |            0.779 |
| recall_T3_moderada    |               0.252 |            0.452 |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': 0.253, 'ic_inf': 0.207, 'ic_sup': 0.293, 'n': 1368}

### LINEA A|holdout

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.052 |            0.052 |
| tasa_alertas_festivos |               0.222 |            0.111 |
| recall_moderada       |               0.702 |            0.9   |
| recall_severa         |               0.945 |            0.986 |
| recall_leve           |               0.258 |            0.582 |
| recall_moderada_media |               0.702 |            0.9   |
| localizacion_hora     |               0.977 |            0.982 |
| recall_T1_moderada    |               0.86  |            0.956 |
| recall_T2_moderada    |               0.772 |            0.915 |
| recall_T3_moderada    |               0.475 |            0.83  |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': 0.198, 'ic_inf': 0.177, 'ic_sup': 0.216, 'n': 1092}

### LINEA B|dev

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.051 |            0.051 |
| tasa_alertas_festivos |               0.087 |            0.043 |
| recall_moderada       |               0.423 |            0.582 |
| recall_severa         |               0.831 |            0.902 |
| recall_leve           |               0.083 |            0.103 |
| recall_moderada_media |               0.423 |            0.582 |
| localizacion_hora     |               0.959 |            0.957 |
| recall_T1_moderada    |               0.701 |            0.769 |
| recall_T2_moderada    |               0.451 |            0.754 |
| recall_T3_moderada    |               0.119 |            0.222 |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': 0.158, 'ic_inf': 0.13, 'ic_sup': 0.189, 'n': 1365}

### LINEA B|holdout

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.052 |            0.052 |
| tasa_alertas_festivos |               0.167 |            0.111 |
| recall_moderada       |               0.638 |            0.691 |
| recall_severa         |               0.918 |            0.941 |
| recall_leve           |               0.181 |            0.197 |
| recall_moderada_media |               0.638 |            0.691 |
| localizacion_hora     |               0.967 |            0.967 |
| recall_T1_moderada    |               0.807 |            0.831 |
| recall_T2_moderada    |               0.713 |            0.771 |
| recall_T3_moderada    |               0.395 |            0.47  |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': 0.052, 'ic_inf': 0.026, 'ic_sup': 0.077, 'n': 1086}

### LINEA 1|dev

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.05  |            0.05  |
| tasa_alertas_festivos |               0.208 |            0.083 |
| recall_moderada       |               0.548 |            0.671 |
| recall_severa         |               0.934 |            0.96  |
| recall_leve           |               0.156 |            0.24  |
| recall_moderada_media |               0.548 |            0.671 |
| localizacion_hora     |               0.972 |            0.968 |
| recall_T1_moderada    |               0.838 |            0.877 |
| recall_T2_moderada    |               0.581 |            0.814 |
| recall_T3_moderada    |               0.224 |            0.322 |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': 0.124, 'ic_inf': 0.094, 'ic_sup': 0.151, 'n': 1368}

### LINEA 1|holdout

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.052 |            0.052 |
| tasa_alertas_festivos |               0.222 |            0.111 |
| recall_moderada       |               0.703 |            0.794 |
| recall_severa         |               0.946 |            0.971 |
| recall_leve           |               0.245 |            0.369 |
| recall_moderada_media |               0.703 |            0.794 |
| localizacion_hora     |               0.971 |            0.975 |
| recall_T1_moderada    |               0.882 |            0.909 |
| recall_T2_moderada    |               0.766 |            0.824 |
| recall_T3_moderada    |               0.462 |            0.648 |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': 0.091, 'ic_inf': 0.067, 'ic_sup': 0.11, 'n': 1092}

### LINEA T-A|dev

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.051 |            0.051 |
| tasa_alertas_festivos |               0.042 |            0.042 |
| recall_moderada       |               0.078 |            0.1   |
| recall_severa         |               0.524 |            0.664 |
| recall_leve           |               0.054 |            0.065 |
| recall_moderada_media |               0.078 |            0.1   |
| localizacion_hora     |               0.904 |            0.917 |
| recall_T1_moderada    |               0.099 |            0.137 |
| recall_T2_moderada    |               0.062 |            0.088 |
| recall_T3_moderada    |               0.073 |            0.075 |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': 0.022, 'ic_inf': 0.007, 'ic_sup': 0.034, 'n': 1362}

### LINEA T-A|holdout

|                       |   perfil_contextual |   coherencia_red |
|:----------------------|--------------------:|-----------------:|
| tasa_alertas          |               0.052 |            0.052 |
| tasa_alertas_festivos |               0.111 |            0.167 |
| recall_moderada       |               0.078 |            0.077 |
| recall_severa         |               0.505 |            0.577 |
| recall_leve           |               0.056 |            0.055 |
| recall_moderada_media |               0.078 |            0.077 |
| localizacion_hora     |               0.875 |            0.879 |
| recall_T1_moderada    |               0.108 |            0.108 |
| recall_T2_moderada    |               0.077 |            0.075 |
| recall_T3_moderada    |               0.05  |            0.05  |

Diferencia pareada de recall moderado (coherencia − perfil): {'diferencia': -0.001, 'ic_inf': -0.011, 'ic_sup': 0.01, 'n': 1086}

## H2 — Alcance local vs. generalizado (inyecciones en el holdout)

Exactitud: **0.884** (n = 4344)

|              |   generalizada |   local |   multilinea |
|:-------------|---------------:|--------:|-------------:|
| generalizada |           1898 |     103 |          171 |
| local        |             43 |    1942 |          187 |

|             |   generalizada |   local |
|:------------|---------------:|--------:|
| T1_moderada |          0.92  |   0.871 |
| T1_severa   |          0.973 |   0.893 |
| T2_moderada |          0.786 |   0.891 |
| T2_severa   |          0.989 |   0.81  |
| T3_moderada |          0.684 |   0.944 |
| T3_severa   |          0.887 |   0.956 |

Días reales del holdout con alguna hora |z| > 3 en la Línea A, por alcance: {'local': 62, 'multilinea': 33, 'generalizada': 11}

- **2023-10-29**: alcance = generalizada (fracción 0.62, caida); acompañan: LINEA B, LINEA K, LINEA M, LINEA P, LINEA T-A
- **2023-08-28**: alcance = multilinea (fracción 0.40, caida); acompañan: LINEA 1, LINEA B, LINEA M, LINEA T-A
- **2023-08-22**: alcance = local (fracción 0.11, aumento); acompañan: LINEA T-A

## H3 — Redistribución el 2023-08-28 (17–19 h)

|           |   h17 |   h18 |   h19 |
|:----------|------:|------:|------:|
| LINEA 1   |  -1.9 |  -6.1 |  -2.6 |
| LINEA 2   |   0.5 |   1.2 |   6.8 |
| LINEA A   | -12.6 | -45.2 | -66.4 |
| LINEA B   |  -1.3 |  -8.7 | nan   |
| LINEA H   |   0.9 |  -1.3 |  -1   |
| LINEA J   |  -1.2 |  -1.8 | nan   |
| LINEA K   |  -1.4 |  -2.8 | nan   |
| LINEA M   |  -2.9 |  -7.5 |  -2.9 |
| LINEA O   |   0.1 |   0.9 |   2.1 |
| LINEA P   |  -0.6 |  -3.8 | nan   |
| LINEA T-A |  -3.9 |  -6.4 |  -6.5 |

Líneas con aumento z > 3: ['LINEA 2']

## H4 — Fallas de datos a nivel de red

- Días reales (2022-07…2024-06) con ≥ 3 líneas habituales sin dato o en cero a la misma hora: 4 de 731 (0.55 %)
- Recall con fallas simuladas en 3–6 líneas: 1.0 (n = 200)
- Alertas de red cuando fallan solo 1–2 líneas (deben ser 0): 0.035 (n = 200)

Alertas reales:

- 2023-08-28 h19: 3 líneas (LINEA B,LINEA J,LINEA K)
- 2023-10-29 h08: 6 líneas (LINEA B,LINEA H,LINEA J,LINEA K,LINEA M,LINEA P)
- 2023-10-29 h09: 5 líneas (LINEA H,LINEA J,LINEA K,LINEA M,LINEA P)
- 2023-10-29 h10: 5 líneas (LINEA H,LINEA J,LINEA K,LINEA M,LINEA P)
- 2023-10-29 h11: 5 líneas (LINEA B,LINEA H,LINEA J,LINEA K,LINEA M)
- 2023-10-29 h12: 5 líneas (LINEA B,LINEA H,LINEA J,LINEA M,LINEA P)
- 2023-10-29 h15: 4 líneas (LINEA B,LINEA J,LINEA M,LINEA P)
- 2023-12-24 h22: 9 líneas (LINEA 1,LINEA 2,LINEA A,LINEA B,LINEA K,LINEA M,LINEA O,LINEA P,LINEA T-A)
- 2023-12-31 h22: 9 líneas (LINEA 1,LINEA 2,LINEA A,LINEA B,LINEA K,LINEA M,LINEA O,LINEA P,LINEA T-A)