# Ficha de los datos

## Origen
Archivos `data/Afluencia_Metro_{2019..2024}.xlsx`: conteos horarios de entradas por línea de servicio del Metro de Medellín. El README original remite al conjunto de Kaggle "afluencia-metro-de-medelln-2023". **No se verificó** la procedencia ni la licencia contra una fuente oficial del Metro (el dominio oficial no es accesible desde el entorno de trabajo). `dev/data/` es una copia idéntica (mismos SHA-256) que usa el notebook legacy.

## Esquema (validado por `metro_anomaly.data.standardize`)
| Columna | Tipo | Notas |
|---|---|---|
| `Día` | texto dd/mm/aaaa (o fecha) | se interpreta día-primero; las fechas inválidas se rechazan |
| `Línea de Servicio` | texto | se normaliza: mayúsculas y sin tildes (`Línea A` → `LINEA A`) |
| `04:00:00` … `23:00:00` | numérico ≥ 0 | 20 columnas horarias; NaN = sin dato (no se imputa) |
| `Total general (Número de pasajeros)` | numérico | igual a Σ horas en > 99,9 % de las filas |

## Cobertura
- 2019-01-01 → 2024-06-30; 11–12 líneas por año; 1 duplicado entre archivos (2024-01-01, Línea B, con valores distintos → se conserva el del archivo de 2024).
- Línea A: 2.008 días; 3 sin registro (2020-06-13, 2020-08-08, 2020-11-12); 18 días de 2019 y 2021 con horas faltantes en 04–21 (huecos de captura); 2022-01 → 2024-06 completo en 04–21.
- 23:00: falta en el 78 % de los domingos y festivos y en el 18 % de los días hábiles; aporta ~0,04 % del flujo de un día hábil → **excluida de la puntuación**.
- 22:00: falta el 24/12 y el 31/12/2023 → alerta de calidad de datos; esos días no se puntúan.

Detalle reproducible: `reports/calidad_datos.md` (`python scripts/data_quality.py`).

## Calendario
- Festivos de Colombia: librería `holidays==0.58`. Contraste empírico: 40/40 festivos entre semana de 2022–2024S1 tienen perfil dominical, y no hay días hábiles con perfil festivo fuera de la lista. **No se contrastó** contra el texto oficial de la Ley 51 de 1983 (red bloqueada).
- Semana Santa: lunes a sábado anteriores a la Pascua (cómputo `dateutil.easter`). Temporada de fin de año: 20/12–10/01. Días especiales: 24/12 y 31/12.
- Jornadas electorales (registraduria.gov.co, vía resultados de búsqueda): 2019-10-27, 2022-03-13, 2022-05-29, 2022-06-19, 2023-10-29. Se usan para **anotar** alertas, no como contexto del modelo.

## Particiones usadas en la evaluación
DEV 2022-01-01…2023-06-30 (selección) · HOLDOUT 2023-07-01…2024-06-30 (evaluación única; ver la bitácora sobre la segunda lectura) · EXTERNO 2019 (transferencia) y 2020–2021 (estrés COVID).

## Eventos reales (`data/eventos_reales.csv`)
7 eventos con fuente. Prospectivos (cuentan para D7): EV1–EV4 (COVID 2020; elecciones 2019 y 2023). A posteriori (no cuentan): EV5–EV7 (elecciones de 2022). No se hallaron incidentes de la Línea A en 2022–2024 con fuente oficial.
