# Determinismo de los artefactos de la plataforma (generado por `scripts/fase3/check_determinism_plataforma.py`)

Dos construcciones completas de `scripts/fase3/build_plataforma.py` en copias aisladas (sin `.git`). Los `.csv.gz` se comparan descomprimidos (el encabezado gzip lleva la fecha) y en el manifiesto se ignoran `creado`, `git_commit_build` y `segundos_construccion`.

| Salida | SHA-256 A | SHA-256 B | A = B | A = versionado |
|---|---|---|---|---|
| `dias.csv.gz` | `819f6809af75f07c` | `819f6809af75f07c` | ✔ | ✔ |
| `horas.csv.gz` | `142b95f593b3a616` | `142b95f593b3a616` | ✔ | ✔ |
| `pronostico_holdout.csv.gz` | `c45ba510d8879092` | `c45ba510d8879092` | ✔ | ✔ |
| `pronostico_siguiente.csv` | `4c7d45443b2bdef6` | `4c7d45443b2bdef6` | ✔ | ✔ |
| `manifiesto.json` | `508aba646e771d96` | `508aba646e771d96` | ✔ | ✔ |

**Resultado: DETERMINISTA ✔** · coincide con los artefactos versionados ✔