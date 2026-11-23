# Determinismo (D9) — generado por `scripts/check_determinism.py`

Dos ejecuciones completas de `scripts/reproduce.py` (sin TensorFlow) en copias aisladas del repositorio. En el artefacto se ignoran los campos volátiles `created_utc` y `git_commit_build`.

| Salida | SHA-256 ejecución A | SHA-256 ejecución B | Idéntico |
|---|---|---|---|
| `configs/seleccion.yaml` | `54b82bf68a3fc61e` | `54b82bf68a3fc61e` | ✔ |
| `reports/seleccion_dev.md` | `e1189ffbe0a35b6c` | `e1189ffbe0a35b6c` | ✔ |
| `reports/holdout.md` | `287ebac86ecef044` | `287ebac86ecef044` | ✔ |
| `reports/holdout_metricas.csv` | `359c16ebf249ecfd` | `359c16ebf249ecfd` | ✔ |
| `reports/externo.md` | `3a19e00e409de887` | `3a19e00e409de887` | ✔ |
| `reports/calidad_datos.md` | `9bca8a56a6879f6e` | `9bca8a56a6879f6e` | ✔ |
| `reports/etiquetas.md` | `7ca40272475310cf` | `7ca40272475310cf` | ✔ |
| `artifacts/modelo.json` | `f2a0ca9bbfdeb408` | `f2a0ca9bbfdeb408` | ✔ |

**Resultado: DETERMINISTA ✔**