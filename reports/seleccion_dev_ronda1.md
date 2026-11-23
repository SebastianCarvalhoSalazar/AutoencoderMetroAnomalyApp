# Selección de modelo en DEV (generado por `scripts/select_model.py`)

Periodo evaluado: 2022-04-01…2023-06-30 (rolling mensual, ajuste desde 2022-01-01). Umbral por configuración: cuantil 95 % de sus scores de días reales (tasa de alertas = 5 % por construcción). IC 95 %: bootstrap por semanas (semilla 0).

| detector             | agg   |   n_semillas |   recall_moderada_media |   recall_mod_ic_inf |   recall_mod_ic_sup |   recall_mod_sd_semillas |   tasa_alertas_festivos |   recall_severa |   recall_leve |   recall_T1_moderada |   recall_T2_moderada |   recall_T3_moderada |   recall_T4_cero |   recall_T4_duplicada |   recall_T4_desfase |   recall_T5_perfil_festivo |   localizacion_hora |
|:---------------------|:------|-------------:|------------------------:|--------------------:|--------------------:|-------------------------:|------------------------:|----------------:|--------------:|---------------------:|---------------------:|---------------------:|-----------------:|----------------------:|--------------------:|---------------------------:|--------------------:|
| ae_contextual        | top2  |            1 |                   0.483 |               0.456 |               0.514 |                    0     |                   0.208 |           0.802 |         0.189 |                0.917 |                0.066 |                0.467 |            1     |                 0.838 |               1     |                      0     |               0.921 |
| ae_contextual        | max   |            3 |                   0.426 |               0.425 |               0.474 |                    0.019 |                   0.181 |           0.786 |         0.126 |                0.939 |                0.069 |                0.27  |            1     |                 0.939 |               0.999 |                      0.022 |               0.935 |
| ae_original          | max   |            3 |                   0.346 |               0.322 |               0.392 |                    0.021 |                   0.028 |           0.558 |         0.119 |                0.426 |                0.05  |                0.564 |            0.552 |                 0.481 |               0.849 |                      0     |             nan     |
| perfil_robusto       | max   |            1 |                   0.34  |               0.306 |               0.379 |                    0     |                   0.083 |           0.866 |         0.098 |                0.719 |                0.127 |                0.173 |            1     |                 0.669 |               1     |                      1     |               0.961 |
| naive_semanal        | max   |            1 |                   0.308 |               0.284 |               0.336 |                    0     |                   0.292 |           0.872 |         0.072 |                0.75  |                0.11  |                0.066 |            1     |                 0.669 |               1     |                      1     |               0.947 |
| perfil_robusto       | top2  |            1 |                   0.304 |               0.268 |               0.348 |                    0     |                   0.083 |           0.811 |         0.092 |                0.616 |                0.129 |                0.167 |            1     |                 0.325 |               0.989 |                      1     |               0.955 |
| naive_semanal        | top2  |            1 |                   0.265 |               0.238 |               0.296 |                    0     |                   0.25  |           0.825 |         0.076 |                0.618 |                0.11  |                0.066 |            1     |                 0.257 |               1     |                      1     |               0.942 |
| regresion_calendario | max   |            1 |                   0.227 |               0.195 |               0.259 |                    0     |                   0.208 |           0.842 |         0.051 |                0.445 |                0.061 |                0.173 |            1     |                 0.548 |               0.985 |                      1     |               0.955 |
| regresion_calendario | top2  |            1 |                   0.173 |               0.144 |               0.203 |                    0     |                   0.208 |           0.771 |         0.05  |                0.329 |                0.048 |                0.14  |            1     |                 0.151 |               0.906 |                      1     |               0.949 |
| isolation_forest     | max   |            1 |                   0.167 |               0.128 |               0.213 |                    0     |                   0.292 |           0.32  |         0.085 |                0.09  |                0.333 |                0.077 |            0.09  |                 0.059 |               0.732 |                      0.039 |             nan     |

## Decisión

```yaml
detector: perfil_robusto
agg: max
umbral: 9.987028968095384
presupuesto_alertas: 0.05
procedencia_umbral: cuantil 0,95 de los scores fuera de muestra de los días reales
  de DEV (2022-04-01…2023-06-30), rolling mensual
mejor_por_recall: ae_original/max (0.346)
empatados:
- ae_original/max
- perfil_robusto/max
baselines_umbral:
  ae_contextual/top2: 9.528508684422558
  ae_contextual/max: 11.294387644297686
  ae_original/max: 3578.1333586040296
  perfil_robusto/max: 9.987028968095384
  naive_semanal/max: 11.998833043342628
  perfil_robusto/top2: 9.206665413302918
  naive_semanal/top2: 10.284851927378092
  regresion_calendario/max: 9.826641016406805
  regresion_calendario/top2: 9.433055726531396
  isolation_forest/max: 0.5513176490003855

```