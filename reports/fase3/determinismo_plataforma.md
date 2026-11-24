# Determinismo de los artefactos de la plataforma (generado por `scripts/fase3/check_determinism_plataforma.py`)

Dos construcciones completas de `scripts/fase3/build_plataforma.py` en copias aisladas (sin `.git`). Los `.csv.gz` se comparan descomprimidos (el encabezado gzip lleva la fecha) y en el manifiesto se ignoran `creado`, `git_commit_build` y `segundos_construccion`.

| Salida | SHA-256 A | SHA-256 B | A = B | A = versionado |
|---|---|---|---|---|
| `dias.csv.gz` | `265b34b8a545595a` | `265b34b8a545595a` | ✔ | ✔ |
| `horas.csv.gz` | `574835d50bf70a00` | `574835d50bf70a00` | ✔ | ✔ |
| `pronostico_holdout.csv.gz` | `c45ba510d8879092` | `c45ba510d8879092` | ✔ | ✔ |
| `pronostico_siguiente.csv` | `a65376e0dd21b0be` | `a65376e0dd21b0be` | ✔ | ✔ |
| `manifiesto.json` | `605c01838dfe8991` | `605c01838dfe8991` | ✔ | ✔ |

**Resultado: DETERMINISTA ✔** · coincide con los artefactos versionados ✔