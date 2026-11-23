# Evaluación externa (generado por `scripts/evaluate_external.py`)

Configuración elegida: **perfil_contextual/peak_shift** (umbral DEV 7.3880). Cada configuración usa su propio umbral de DEV.

## 2019 — transferencia (D6)

Ajuste con DEV (2022-01…2023-06); nivel y referencia causales sobre 2019; evaluación 2019-03-01…2019-12-31 (días con las 19 horas completas). Advertencia: no es un backtest causal en el calendario (el modelo de forma se aprende con años posteriores).

| config                       |   tasa_alertas |   tasa_festivos |   recall_severa |   recall_moderada |   recall_T5 | D6_cumple(≤15%)   |
|:-----------------------------|---------------:|----------------:|----------------:|------------------:|------------:|:------------------|
| perfil_contextual/peak_shift |          0.075 |           0.067 |           0.925 |             0.656 |       0.995 | True              |
| perfil_robusto/max           |          0.071 |           0.067 |           0.896 |             0.524 |       1     | True              |
| regresion_contextual/max     |          0.061 |           0.133 |           0.908 |             0.533 |       1     | True              |
| naive_semanal/max            |          0.099 |           0.133 |           0.875 |             0.351 |       1     | True              |
| ae_original/max              |          0.167 |           0.067 |           0.72  |             0.512 |       0     | False             |

Alertas del modelo elegido en 2019 (22):

|            | día       |   score | calendario                                            |
|:-----------|:----------|--------:|:------------------------------------------------------|
| 2019-03-08 | Friday    |   64.87 |                                                       |
| 2019-03-29 | Friday    |    8.4  |                                                       |
| 2019-04-08 | Monday    |   38.07 |                                                       |
| 2019-04-25 | Thursday  |  150.66 |                                                       |
| 2019-04-27 | Saturday  |   27.26 |                                                       |
| 2019-05-17 | Friday    |    8.08 |                                                       |
| 2019-08-10 | Saturday  |    9.11 |                                                       |
| 2019-08-21 | Wednesday |   13.5  |                                                       |
| 2019-08-22 | Thursday  |    8.43 |                                                       |
| 2019-09-04 | Wednesday |  151.65 |                                                       |
| 2019-09-16 | Monday    |  110.46 |                                                       |
| 2019-10-23 | Wednesday |  111.7  |                                                       |
| 2019-10-27 | Sunday    |   57.63 | Elecciones de autoridades locales                     |
| 2019-11-12 | Tuesday   |    9.56 |                                                       |
| 2019-11-21 | Thursday  |   40.57 |                                                       |
| 2019-11-27 | Wednesday |   11.89 |                                                       |
| 2019-12-17 | Tuesday   |    8.43 |                                                       |
| 2019-12-18 | Wednesday |    8.35 |                                                       |
| 2019-12-19 | Thursday  |    7.89 |                                                       |
| 2019-12-24 | Tuesday   |   10.58 | Temporada de fin de año; Víspera de Navidad/Año Nuevo |
| 2019-12-25 | Wednesday |   50.97 | Navidad; Temporada de fin de año                      |
| 2019-12-30 | Monday    |   79.06 | Temporada de fin de año                               |

## 2020–2021 — estrés COVID (causal, rolling mensual desde 2019)

Tasa de alertas mensual:

| mes     |   perfil_contextual/peak_shift |   perfil_robusto/max |   regresion_contextual/max |   naive_semanal/max |   ae_original/max |
|:--------|-------------------------------:|---------------------:|---------------------------:|--------------------:|------------------:|
| 2020-01 |                           0.06 |                 0.13 |                       0.06 |                0.1  |              0.1  |
| 2020-02 |                           0.07 |                 0.07 |                       0.07 |                0.07 |              0.07 |
| 2020-03 |                           0.52 |                 0.52 |                       0.48 |                0.23 |              0.52 |
| 2020-04 |                           0.9  |                 1    |                       0.83 |                0    |              1    |
| 2020-05 |                           0.9  |                 1    |                       0.42 |                0.19 |              1    |
| 2020-06 |                           0.45 |                 0.83 |                       0    |                0    |              1    |
| 2020-07 |                           0.19 |                 0.26 |                       0.13 |                0.23 |              0.68 |
| 2020-08 |                           0.1  |                 0.1  |                       0    |                0.23 |              0.73 |
| 2020-09 |                           0    |                 0    |                       0    |                0    |              0.87 |
| 2020-10 |                           0    |                 0    |                       0    |                0    |              0.84 |
| 2020-11 |                           0    |                 0    |                       0    |                0    |              0.69 |
| 2020-12 |                           0.03 |                 0.03 |                       0.03 |                0.1  |              0.21 |
| 2021-01 |                           0.03 |                 0.03 |                       0.07 |                0.24 |              0.1  |
| 2021-02 |                           0    |                 0    |                       0    |                0    |              0.14 |
| 2021-03 |                           0    |                 0.03 |                       0    |                0.03 |              0.16 |
| 2021-04 |                           0.2  |                 0.17 |                       0.03 |                0.47 |              0.2  |
| 2021-05 |                           0    |                 0    |                       0.03 |                0.17 |              0.1  |
| 2021-06 |                           0    |                 0    |                       0    |                0.07 |              0.17 |
| 2021-07 |                           0    |                 0    |                       0    |                0    |              0.16 |
| 2021-08 |                           0    |                 0    |                       0    |                0.06 |              0    |
| 2021-09 |                           0    |                 0    |                       0    |                0    |              0.07 |
| 2021-10 |                           0.03 |                 0.03 |                       0    |                0    |              0.32 |
| 2021-11 |                           0.03 |                 0.03 |                       0    |                0.07 |              0.69 |
| 2021-12 |                           0.1  |                 0.23 |                       0    |                0.13 |              0.19 |

## D7 — eventos reales prospectivos con fuente oficial

Detección = score > umbral en la fecha de inicio o en los 2 días siguientes.

| evento   | fecha      | periodo   | descripción                                                            | perfil_contextual/peak_shift   | perfil_robusto/max       | regresion_contextual/max   | naive_semanal/max        | ae_original/max           |
|:---------|:-----------|:----------|:-----------------------------------------------------------------------|:-------------------------------|:-------------------------|:---------------------------|:-------------------------|:--------------------------|
| EV1      | 2020-03-20 | covid     | Decreto municipal 0392 del 20/03/2020 (Medellín): medidas especiales e | ✔ 2020-03-20 (max 81.4)        | ✔ 2020-03-20 (max 64.6)  | ✔ 2020-03-20 (max 40.3)    | ✔ 2020-03-20 (max 52.7)  | ✔ 2020-03-20 (max 8279.0) |
| EV2      | 2020-03-25 | covid     | Decreto 457 del 22/03/2020: aislamiento preventivo obligatorio naciona | ✔ 2020-03-25 (max 78.9)        | ✔ 2020-03-25 (max 69.7)  | ✔ 2020-03-25 (max 54.5)    | ✔ 2020-03-25 (max 25.2)  | ✔ 2020-03-25 (max 6496.3) |
| EV3      | 2019-10-27 | ext2019   | Elecciones de autoridades locales (último domingo de octubre)          | ✔ 2019-10-27 (max 57.6)        | ✔ 2019-10-27 (max 55.9)  | ✔ 2019-10-27 (max 42.5)    | ✔ 2019-10-27 (max 115.7) | ✔ 2019-10-27 (max 4359.8) |
| EV4      | 2023-10-29 | holdout   | Elecciones territoriales 2023                                          | ✔ 2023-10-29 (max 101.0)       | ✔ 2023-10-29 (max 109.1) | ✔ 2023-10-29 (max 61.2)    | ✔ 2023-10-29 (max 198.6) | ✔ 2023-10-29 (max 5095.7) |

**Modelo elegido: 4/4 eventos detectados** (meta ≥ 2/3 → ✔).

Scores diarios del modelo elegido alrededor de EV1/EV2 (marzo y abril de 2020):

| date                |   score | alerta   |
|:--------------------|--------:|:---------|
| 2020-03-10 00:00:00 |    5.51 | False    |
| 2020-03-11 00:00:00 |    3.58 | False    |
| 2020-03-12 00:00:00 |    3.41 | False    |
| 2020-03-13 00:00:00 |    3.15 | False    |
| 2020-03-14 00:00:00 |    5.45 | False    |
| 2020-03-15 00:00:00 |    4.59 | False    |
| 2020-03-16 00:00:00 |   11.67 | True     |
| 2020-03-17 00:00:00 |   17.28 | True     |
| 2020-03-18 00:00:00 |   22.18 | True     |
| 2020-03-19 00:00:00 |   23.92 | True     |
| 2020-03-20 00:00:00 |   29.08 | True     |
| 2020-03-21 00:00:00 |   81.41 | True     |
| 2020-03-22 00:00:00 |   34.88 | True     |
| 2020-03-23 00:00:00 |   31.2  | True     |
| 2020-03-24 00:00:00 |   55.29 | True     |
| 2020-03-25 00:00:00 |   77.74 | True     |
| 2020-03-26 00:00:00 |   78.92 | True     |
| 2020-03-27 00:00:00 |   73.59 | True     |
| 2020-03-28 00:00:00 |   79.14 | True     |
| 2020-03-29 00:00:00 |   34.92 | True     |
| 2020-03-30 00:00:00 |   67.77 | True     |
| 2020-03-31 00:00:00 |   67    | True     |
| 2020-04-01 00:00:00 |   59.69 | True     |
| 2020-04-02 00:00:00 |   57.59 | True     |
| 2020-04-03 00:00:00 |   54.23 | True     |
| 2020-04-04 00:00:00 |   55    | True     |
| 2020-04-05 00:00:00 |   11.25 | True     |
| 2020-04-06 00:00:00 |    8.21 | True     |
| 2020-04-07 00:00:00 |    4.59 | False    |
| 2020-04-08 00:00:00 |    4.07 | False    |
| 2020-04-09 00:00:00 |    9.64 | True     |
| 2020-04-10 00:00:00 |    8.72 | True     |
| 2020-04-11 00:00:00 |    5.74 | False    |
| 2020-04-12 00:00:00 |    8.54 | True     |
| 2020-04-13 00:00:00 |   13.57 | True     |
| 2020-04-14 00:00:00 |   13.85 | True     |
| 2020-04-15 00:00:00 |   15.06 | True     |
