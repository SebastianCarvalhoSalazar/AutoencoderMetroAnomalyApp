# Evaluación externa (generado por `scripts/evaluate_external.py`)

Configuración elegida: **perfil_contextual/top2** (umbral DEV 7.3482). Cada configuración usa su propio umbral de DEV.

## 2019 — transferencia (D6)

Ajuste con DEV (2022-01…2023-06); nivel y referencia causales sobre 2019; evaluación 2019-03-01…2019-12-31 (días con las 19 horas completas). Advertencia: no es un backtest causal en el calendario (el modelo de forma se aprende con años posteriores).

| config                   |   tasa_alertas |   tasa_festivos |   recall_severa |   recall_moderada |   recall_T5 | D6_cumple(≤15%)   |
|:-------------------------|---------------:|----------------:|----------------:|------------------:|------------:|:------------------|
| perfil_contextual/top2   |          0.109 |           0.2   |           0.942 |             0.728 |           1 | True              |
| perfil_robusto/max       |          0.071 |           0.067 |           0.896 |             0.524 |           1 | True              |
| regresion_contextual/max |          0.041 |           0.067 |           0.874 |             0.374 |           1 | True              |
| naive_semanal/max        |          0.099 |           0.133 |           0.875 |             0.351 |           1 | True              |
| ae_original/max          |          0.167 |           0.067 |           0.72  |             0.512 |           0 | False             |

Alertas del modelo elegido en 2019 (32):

|            | día       |   score | calendario                                            |
|:-----------|:----------|--------:|:------------------------------------------------------|
| 2019-03-08 | Friday    |   47.02 |                                                       |
| 2019-04-08 | Monday    |   24.6  |                                                       |
| 2019-04-25 | Thursday  |  127.24 |                                                       |
| 2019-04-27 | Saturday  |   26.66 |                                                       |
| 2019-05-17 | Friday    |    9.7  |                                                       |
| 2019-07-20 | Saturday  |    7.87 | Día de la Independencia                               |
| 2019-07-24 | Wednesday |    8.66 |                                                       |
| 2019-08-10 | Saturday  |   11.06 |                                                       |
| 2019-08-21 | Wednesday |   15.43 |                                                       |
| 2019-08-22 | Thursday  |    7.46 |                                                       |
| 2019-09-04 | Wednesday |  126.08 |                                                       |
| 2019-09-16 | Monday    |   66.2  |                                                       |
| 2019-09-24 | Tuesday   |    7.85 |                                                       |
| 2019-10-23 | Wednesday |   70.87 |                                                       |
| 2019-10-27 | Sunday    |   57.46 | Elecciones de autoridades locales                     |
| 2019-11-12 | Tuesday   |    9.83 |                                                       |
| 2019-11-21 | Thursday  |   38.24 |                                                       |
| 2019-11-27 | Wednesday |   11.6  |                                                       |
| 2019-12-03 | Tuesday   |    7.39 |                                                       |
| 2019-12-05 | Thursday  |    7.64 |                                                       |
| 2019-12-08 | Sunday    |    7.97 | La Inmaculada Concepción                              |
| 2019-12-13 | Friday    |    7.48 |                                                       |
| 2019-12-14 | Saturday  |    8.12 |                                                       |
| 2019-12-15 | Sunday    |    7.44 |                                                       |
| 2019-12-16 | Monday    |    8.25 |                                                       |
| 2019-12-17 | Tuesday   |    8.67 |                                                       |
| 2019-12-18 | Wednesday |    8.59 |                                                       |
| 2019-12-19 | Thursday  |    8.57 |                                                       |
| 2019-12-21 | Saturday  |    7.64 | Temporada de fin de año                               |
| 2019-12-24 | Tuesday   |    8.92 | Temporada de fin de año; Víspera de Navidad/Año Nuevo |
| 2019-12-25 | Wednesday |   31.67 | Navidad; Temporada de fin de año                      |
| 2019-12-30 | Monday    |   27.33 | Temporada de fin de año                               |

## 2020–2021 — estrés COVID (causal, rolling mensual desde 2019)

Tasa de alertas mensual:

| mes     |   perfil_contextual/top2 |   perfil_robusto/max |   regresion_contextual/max |   naive_semanal/max |   ae_original/max |
|:--------|-------------------------:|---------------------:|---------------------------:|--------------------:|------------------:|
| 2020-01 |                     0.03 |                 0.13 |                       0.03 |                0.1  |              0.1  |
| 2020-02 |                     0.07 |                 0.07 |                       0.07 |                0.07 |              0.07 |
| 2020-03 |                     0.52 |                 0.52 |                       0.48 |                0.23 |              0.52 |
| 2020-04 |                     0.97 |                 1    |                       0.43 |                0    |              1    |
| 2020-05 |                     1    |                 1    |                       0.19 |                0.19 |              1    |
| 2020-06 |                     0.86 |                 0.83 |                       0    |                0    |              1    |
| 2020-07 |                     0.61 |                 0.26 |                       0    |                0.23 |              0.68 |
| 2020-08 |                     0.13 |                 0.1  |                       0    |                0.23 |              0.73 |
| 2020-09 |                     0    |                 0    |                       0    |                0    |              0.87 |
| 2020-10 |                     0    |                 0    |                       0    |                0    |              0.84 |
| 2020-11 |                     0    |                 0    |                       0    |                0    |              0.69 |
| 2020-12 |                     0.03 |                 0.03 |                       0    |                0.1  |              0.21 |
| 2021-01 |                     0    |                 0.03 |                       0    |                0.24 |              0.1  |
| 2021-02 |                     0    |                 0    |                       0    |                0    |              0.14 |
| 2021-03 |                     0    |                 0.03 |                       0    |                0.03 |              0.16 |
| 2021-04 |                     0.37 |                 0.17 |                       0    |                0.47 |              0.2  |
| 2021-05 |                     0.03 |                 0    |                       0    |                0.17 |              0.1  |
| 2021-06 |                     0    |                 0    |                       0    |                0.07 |              0.17 |
| 2021-07 |                     0    |                 0    |                       0    |                0    |              0.16 |
| 2021-08 |                     0    |                 0    |                       0    |                0.06 |              0    |
| 2021-09 |                     0    |                 0    |                       0    |                0    |              0.07 |
| 2021-10 |                     0.03 |                 0.03 |                       0    |                0    |              0.32 |
| 2021-11 |                     0.03 |                 0.03 |                       0    |                0.07 |              0.69 |
| 2021-12 |                     0.13 |                 0.23 |                       0    |                0.13 |              0.19 |

## D7 — eventos reales prospectivos con fuente oficial

Detección = score > umbral en la fecha de inicio o en los 2 días siguientes.

| evento   | fecha      | periodo   | descripción                                                            | perfil_contextual/top2   | perfil_robusto/max       | regresion_contextual/max   | naive_semanal/max        | ae_original/max           |
|:---------|:-----------|:----------|:-----------------------------------------------------------------------|:-------------------------|:-------------------------|:---------------------------|:-------------------------|:--------------------------|
| EV1      | 2020-03-20 | covid     | Decreto municipal 0392 del 20/03/2020 (Medellín): medidas especiales e | ✔ 2020-03-20 (max 84.4)  | ✔ 2020-03-20 (max 64.6)  | ✔ 2020-03-20 (max 41.8)    | ✔ 2020-03-20 (max 52.7)  | ✔ 2020-03-20 (max 8279.0) |
| EV2      | 2020-03-25 | covid     | Decreto 457 del 22/03/2020: aislamiento preventivo obligatorio naciona | ✔ 2020-03-25 (max 76.6)  | ✔ 2020-03-25 (max 69.7)  | ✔ 2020-03-25 (max 49.4)    | ✔ 2020-03-25 (max 25.2)  | ✔ 2020-03-25 (max 6496.3) |
| EV3      | 2019-10-27 | ext2019   | Elecciones de autoridades locales (último domingo de octubre)          | ✔ 2019-10-27 (max 57.5)  | ✔ 2019-10-27 (max 55.9)  | ✔ 2019-10-27 (max 48.7)    | ✔ 2019-10-27 (max 115.7) | ✔ 2019-10-27 (max 4359.8) |
| EV4      | 2023-10-29 | holdout   | Elecciones territoriales 2023                                          | ✔ 2023-10-29 (max 107.1) | ✔ 2023-10-29 (max 109.1) | ✔ 2023-10-29 (max 66.5)    | ✔ 2023-10-29 (max 198.6) | ✔ 2023-10-29 (max 5095.7) |

**Modelo elegido: 4/4 eventos detectados** (meta ≥ 2/3 → ✔).

Scores diarios del modelo elegido alrededor de EV1/EV2 (marzo y abril de 2020):

| date                |   score | alerta   |
|:--------------------|--------:|:---------|
| 2020-03-10 00:00:00 |    7.08 | False    |
| 2020-03-11 00:00:00 |    5.63 | False    |
| 2020-03-12 00:00:00 |    5.31 | False    |
| 2020-03-13 00:00:00 |    5.39 | False    |
| 2020-03-14 00:00:00 |    6.32 | False    |
| 2020-03-15 00:00:00 |    4.94 | False    |
| 2020-03-16 00:00:00 |   12.25 | True     |
| 2020-03-17 00:00:00 |   17.98 | True     |
| 2020-03-18 00:00:00 |   22.02 | True     |
| 2020-03-19 00:00:00 |   25.26 | True     |
| 2020-03-20 00:00:00 |   29.86 | True     |
| 2020-03-21 00:00:00 |   84.4  | True     |
| 2020-03-22 00:00:00 |   34.2  | True     |
| 2020-03-23 00:00:00 |   29.28 | True     |
| 2020-03-24 00:00:00 |   62.04 | True     |
| 2020-03-25 00:00:00 |   75.83 | True     |
| 2020-03-26 00:00:00 |   76.55 | True     |
| 2020-03-27 00:00:00 |   71.61 | True     |
| 2020-03-28 00:00:00 |   82.34 | True     |
| 2020-03-29 00:00:00 |   35.57 | True     |
| 2020-03-30 00:00:00 |   66.63 | True     |
| 2020-03-31 00:00:00 |   64.9  | True     |
| 2020-04-01 00:00:00 |   56.23 | True     |
| 2020-04-02 00:00:00 |   54.32 | True     |
| 2020-04-03 00:00:00 |   51.61 | True     |
| 2020-04-04 00:00:00 |   54.09 | True     |
| 2020-04-05 00:00:00 |   12.56 | True     |
| 2020-04-06 00:00:00 |   10.36 | True     |
| 2020-04-07 00:00:00 |    7.36 | True     |
| 2020-04-08 00:00:00 |    6.86 | False    |
| 2020-04-09 00:00:00 |   15.78 | True     |
| 2020-04-10 00:00:00 |   13.74 | True     |
| 2020-04-11 00:00:00 |    8.55 | True     |
| 2020-04-12 00:00:00 |   13.62 | True     |
| 2020-04-13 00:00:00 |   18.99 | True     |
| 2020-04-14 00:00:00 |   19.21 | True     |
| 2020-04-15 00:00:00 |   20.25 | True     |
