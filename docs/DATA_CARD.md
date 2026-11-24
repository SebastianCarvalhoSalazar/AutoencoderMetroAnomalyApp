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

## Qué mide el dato (hallazgo F8)
Los conteos corresponden a **entradas registradas por validación de pago**, no a pasajeros físicos. En jornadas electorales el Metro no cobra el pasaje (comunicado oficial, desde 2006; el 29/10/2023 hasta las 17:00) y las entradas aparecen ≈ 0 durante esas horas en todas las líneas. En 2019 hay días en que las horas faltantes se **acumularon en la hora siguiente** (p. ej., 16/02/2019 a las 15 h: 254.265 entradas).

## Calendario
- Festivos de Colombia: librería `holidays==0.58`. Contraste empírico: 40/40 festivos entre semana de 2022–2024S1 tienen perfil dominical, y no hay días hábiles con perfil festivo fuera de la lista. **No se contrastó** contra el texto oficial de la Ley 51 de 1983 (red bloqueada).
- Semana Santa: lunes a sábado anteriores a la Pascua (cómputo `dateutil.easter`). Temporada de fin de año: 20/12–10/01. Días especiales: 24/12 y 31/12.
- Jornadas electorales (registraduria.gov.co, vía resultados de búsqueda): 2019-10-27, 2022-03-13, 2022-05-29, 2022-06-19, 2023-10-29. Se usan para **anotar** alertas, no como contexto del modelo.

## Particiones usadas en la evaluación
DEV 2022-01-01…2023-06-30 (selección) · HOLDOUT 2023-07-01…2024-06-30 (evaluación única; ver la bitácora sobre la segunda lectura) · EXTERNO 2019 (transferencia) y 2020–2021 (estrés COVID).

## Eventos reales (`data/eventos_reales.csv`)
Hay 14 eventos registrados, cada uno con su fuente y su nivel de verificación.
- **EV1–EV7:** COVID 2020 y elecciones de 2019, 2022 y 2023. Los prospectivos para D7 son EV1–EV4.
- **IN1–IN3:** incidentes operacionales de 2023 (Telemedellín citando a @metrodemedellin). Solo IN3 (suspensión del 28/08/2023) deja señal en los datos línea × hora.
- **MED1:** no cobro electoral del 29/10/2023, según el comunicado oficial del Metro.
- **OP1–OP3 (fase 3):** cierre anticipado del 24 y 31 de diciembre.
  - 22:00 según fuentes oficiales: Telemedellín citando al Metro en 2022 y el comunicado N.° 2152 del Metro en 2024.
  - **2023: solo prensa** (21:00). Es coherente con los datos: h21 reducida y h22 sin registro en todas las líneas.

## Datos multilínea y fuentes externas (fase 3)
- **Líneas modeladas:** 11 (A, B, T-A, H, J, K, M, P, 1, 2 y O).
  - La L (cable turístico) no opera todas las franjas y se excluye.
  - En 2023, la A concentra el 63,7 % de las entradas registradas.
  - Los cables tienen días incompletos frecuentes: unos 50 al año, que se marcan como "no evaluable".
- **Topología:** `data/topologia_red.csv` (14 conexiones entre líneas en estaciones de integración) y `data/lineas.csv` (modo, extremos, km y estaciones). Cada fila tiene la URL de la página oficial del Metro. El esquema de la app **no es geográfico**.
- **Clima:** `data/clima_open_meteo_medellin.csv`, lluvia y temperatura horarias 2019–2024 de Open-Meteo.
  - Reanálisis con licencia CC BY 4.0, una sola celda (6,22; −75,55).
  - La precipitación es la suma de la hora precedente, así que la franja h toma el registro h+1.
  - No mejoró el pronóstico (H5b descartada). Se conserva para investigación.
- **Fuentes evaluadas y descartadas:** demografía por comuna (DANE/Alcaldía), encuesta Origen-Destino (AMVA), amenaza por movimientos en masa y corredores del POT. Son anuales o estáticas, o no alineables con línea × hora. Ver `docs/fase3/FUENTES_EXTERNAS.md`.
- **Cierres programados** (`data/cierres_programados.csv`): mantenimientos anuales de los metrocables anunciados por Telemedellín.
  - 2022: P del 20 al 28/06, K del 29/06 al 09/07, M del 29/06 al 01/07.
  - 2024: P del 08 al 13/06, M del 14 al 16/06, H del 17 al 19/06.
  - Para 2023 no se halló un anuncio con año explícito. Los días sin registro de esos cierres se marcan como "línea sin registro".
  - En los datos, una línea cerrada aparece sin filas (no con ceros) o con un registro residual (P en 2024: 300–680 entradas al día).
- **Datos faltantes** para capacidades no viables: ver `docs/fase3/HOJA_DE_RUTA_DATOS.md`.
