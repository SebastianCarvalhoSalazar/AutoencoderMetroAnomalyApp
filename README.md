# Detección de anomalías en la afluencia horaria — Línea A, Metro de Medellín

![Logo](./assets/logo.png)

Sistema que señala **días y horas cuya afluencia se desvía de lo esperado para su contexto de calendario** (día de la semana, festivo, Semana Santa, temporada de fin de año y nivel reciente de demanda), y explica qué horas se desvían, en qué dirección y con qué magnitud. Incluye una aplicación Streamlit, un pipeline reproducible y una evaluación con protocolo pre-registrado.

> **Estado (2026-11-23):** funcional y validado de extremo a extremo, pero **no cumple todos los criterios pre-registrados**. Cumple la tasa de alertas, la detección de perfiles incoherentes con el calendario, la superioridad frente al sistema original, la generalización a 2019 y la detección de eventos reales. **No cumple** la sensibilidad mínima a anomalías moderadas y severas ni la tasa máxima de alertas en festivos. Detalle en [`docs/validacion/INFORME_FINAL.md`](docs/validacion/INFORME_FINAL.md).

## Inicio rápido
```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10
make install            # pip install -r requirements-dev.txt && pip install -e .
make app                # streamlit run app.py
make test               # tests (unitarios, integración, app, contraste de auditoría) + cobertura ≥ 80 %
make reproduce          # pipeline completo: datos → selección → holdout → externo → artefacto (~10 min)
make determinism        # D9: dos ejecuciones aisladas con hashes idénticos
make install-research && make reproduce-research    # incluye las comparaciones con autoencoders (TensorFlow)
```

## Cómo funciona
1. **Datos** (`src/metro_anomaly/data.py`): valida el esquema de los archivos oficiales, normaliza las líneas, resuelve duplicados y **no imputa** (los faltantes se reportan como problema de calidad de datos). Se puntúan las horas 04–22; las 23:00 se excluyen (ver la ficha de datos).
2. **Calendario** (`calendar.py`): festivos de Colombia, Semana Santa, temporada de fin de año, vísperas y jornadas electorales (estas últimas solo para anotar).
3. **Modelo** (`detectors.py`, `perfil_contextual`):
   - perfil robusto (mediana/MAD) por tipo de día y hora, en log1p, relativo a un **nivel causal** (solo usa días anteriores);
   - z por hora;
   - score diario `peak_shift`, que capta tanto picos localizados como desplazamientos de todo el día.
4. **Umbral**: cuantil 95 % de los scores fuera de muestra de DEV (presupuesto de alertas del 5 %), congelado en `artifacts/modelo.json`.
5. **App** (`app.py`, `service.py`): explorar cualquier fecha del histórico o cargar un `.xlsx`/`.csv` con el formato oficial (uno o varios días; para fechas futuras se necesitan ≥ 4 semanas previas en el archivo).

## Validación (resumen)
Protocolo:
- DEV (2022-01…2023-06) para seleccionar el modelo, con rolling origin mensual;
- HOLDOUT (2023-07…2024-06) con modelo y umbral congelados;
- EXTERNO: 2019 y estrés COVID 2020–2021.

La verdad de terreno son **anomalías sintéticas** con un catálogo pre-registrado y **eventos reales con fuente oficial**. Las etiquetas heurísticas del proyecto original **no** se usan como verdad: se demostró que reflejan sobre todo el calendario.

| Criterio | Resultado | |
|---|---|---|
| Tasa de alertas (holdout) 2–10 % | 2,2 % | ✔ |
| Recall severas ≥ 0,95 / moderadas ≥ 0,80 | 0,894 / 0,506 | ✘ / ✘ |
| Alertas en festivos ≤ 10 % | 11,1 % (Navidad y Año Nuevo) | ✘ |
| Perfil de festivo en día hábil ≥ 0,95 | 1,00 | ✔ |
| Mejor que el autoencoder original | +0,207 de recall moderado (IC > 0) | ✔ |
| Transferencia 2019 ≤ 15 % de alertas | 7,5 % | ✔ |
| Eventos reales prospectivos (COVID 2020, elecciones 2019 y 2023) | 4/4 | ✔ |
| Causalidad y determinismo | tests + doble ejecución | ✔ ([`reports/determinismo.md`](reports/determinismo.md)) |

¿Por qué no un autoencoder? Con el protocolo correcto, el AE original queda por debajo en todos los criterios y no detecta perfiles de festivo en días hábiles. El AE contextual tampoco los detecta y genera demasiadas alertas en festivos ([`reports/seleccion_dev.md`](reports/seleccion_dev.md)).

## Limitaciones principales
- Sensibilidad ~50 % a anomalías moderadas, la más débil en aumentos por evento.
- Falsos positivos en Navidad y Año Nuevo.
- Readaptación lenta ante cambios de régimen.
- Solo Línea A.
- La validación primaria es sintética: faltan incidentes operativos verificados. Ver [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Estructura
```
app.py                  App Streamlit (nueva)
src/metro_anomaly/      data, calendar, detectors(_nn), injection, labels, evaluation, artifacts, service
scripts/                data_quality, label_quality, run_eval, select_model, evaluate_holdout, evaluate_external,
                        build_artifact, reproduce, check_determinism
tests/                  unit/, integration/, e2e/, audit/ (contraste independiente de la auditoría)
artifacts/modelo.json   modelo final + umbral + procedencia
configs/seleccion.yaml  selección congelada (y umbrales de las líneas base)
reports/                calidad de datos, etiquetas, selección DEV, holdout, externo, determinismo
data/                   archivos oficiales 2019–2024 + eventos_reales.csv
legacy/                 app original (solo referencia; ver legacy/README.md)
dev/                    notebook y artefactos originales (solo referencia de la auditoría)
audit/                  auditoría técnica y sus experimentos
docs/                   PLAN_MAESTRO, BITACORA, MODEL_CARD, DATA_CARD, validacion/
```

## Trazabilidad
[Auditoría](audit/AUDITORIA.md) → [Plan maestro](docs/PLAN_MAESTRO.md) → [Bitácora](docs/BITACORA.md) → informes por fase en [`docs/validacion/`](docs/validacion/).

## Datos
Archivos de afluencia en `data/`. Según el README original provienen de Kaggle: https://www.kaggle.com/datasets/roysandoval/afluencia-metro-de-medelln-2023 (procedencia y licencia no verificadas contra una fuente oficial del Metro).
