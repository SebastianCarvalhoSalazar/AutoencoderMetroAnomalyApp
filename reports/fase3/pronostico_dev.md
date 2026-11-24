# Pronóstico día siguiente por línea y franja — periodo dev (generado por `scripts/fase3/exp_pronostico.py`)

Origen rodante mensual 2022-07…2023-06; ajuste con 2022-01…M-1. WAPE = Σ|y−ŷ| / Σy (menor es mejor). Métricas sobre las 69649 filas (línea × día × franja) con predicción de todos los modelos (de 74339).

WAPE global: {'naive': 0.0623, 'perfil': 0.0608, 'gbm': 0.0684, 'gbm_v2': 0.0617, 'combo': 0.056, 'combo_v2': 0.0534}

|           |   naive |   perfil |    gbm |   gbm_v2 |   combo |   combo_v2 |   mejora_gbm_vs_naive_% |   mejora_gbm_v2_vs_naive_% |   cobertura_80 |   cobertura_80_calibrada |
|:----------|--------:|---------:|-------:|---------:|--------:|-----------:|------------------------:|---------------------------:|---------------:|-------------------------:|
| LINEA 1   |  0.0608 |   0.0621 | 0.0605 |   0.0585 |  0.0529 |     0.0521 |                     0.5 |                        3.8 |          0.695 |                    0.9   |
| LINEA 2   |  0.084  |   0.0758 | 0.0814 |   0.0794 |  0.07   |     0.0692 |                     3.1 |                        5.5 |          0.702 |                    0.922 |
| LINEA A   |  0.056  |   0.0546 | 0.0641 |   0.0571 |  0.0512 |     0.0485 |                   -14.5 |                       -2   |          0.688 |                    0.905 |
| LINEA B   |  0.0749 |   0.0658 | 0.0779 |   0.0697 |  0.0628 |     0.0597 |                    -4   |                        6.9 |          0.699 |                    0.908 |
| LINEA H   |  0.1175 |   0.1072 | 0.1086 |   0.1079 |  0.1004 |     0.1003 |                     7.6 |                        8.2 |          0.741 |                    0.9   |
| LINEA J   |  0.0757 |   0.0697 | 0.0777 |   0.071  |  0.066  |     0.0629 |                    -2.6 |                        6.2 |          0.721 |                    0.908 |
| LINEA K   |  0.0753 |   0.0665 | 0.089  |   0.0717 |  0.0702 |     0.0632 |                   -18.2 |                        4.8 |          0.735 |                    0.899 |
| LINEA M   |  0.0946 |   0.0796 | 0.0894 |   0.0905 |  0.0771 |     0.0776 |                     5.5 |                        4.3 |          0.76  |                    0.933 |
| LINEA O   |  0.1042 |   0.1274 | 0.1026 |   0.0959 |  0.1005 |     0.0982 |                     1.5 |                        8   |          0.731 |                    0.929 |
| LINEA P   |  0.0788 |   0.0928 | 0.0953 |   0.0796 |  0.0815 |     0.0757 |                   -20.9 |                       -1   |          0.75  |                    0.89  |
| LINEA T-A |  0.076  |   0.0764 | 0.0792 |   0.0737 |  0.067  |     0.0653 |                    -4.2 |                        3   |          0.721 |                    0.91  |

`cobertura_80_calibrada`: banda recalibrada en línea con los meses anteriores (desde el 2.º mes; ajuste añadido después de ver la cobertura del holdout v1).
