# Hoja de ruta de datos — capacidades que requieren nuevas fuentes (F10.7)

Base actual: entradas registradas (validaciones pagas) por **línea × hora**, exportadas por día, 2019-01…2024-06, 12 líneas. A esto se suma lluvia horaria de reanálisis para una celda, que no aportó (H5b descartada).

## 1. Qué es viable hoy y qué no

| Capacidad | Viable con los datos actuales | Evidencia o motivo |
|---|---|---|
| Revisión retrospectiva de días atípicos por línea | **Sí** | Perfil contextual aplicado a 11 líneas |
| Separar anomalía propia de la línea de lo que mueve a toda la red | **Sí** (A, B y 1) | H1: +0,198 de recall moderado en la Línea A con el mismo presupuesto de alertas |
| Alcance local / varias líneas / generalizado | **Orientativo** | H2: exactitud 0,884 (< 0,90) |
| Ausencia simultánea de registro en varias líneas | **Sí** | H4. En 2022–2024 los 4 casos reales fueron operación o calendario, no fallas de recaudo |
| Pronóstico del día siguiente por línea y hora | Ver `reports/fase3/pronostico_holdout.md` | H5 |
| Anticipar días de alta demanda | Parcial | H6: mejor que el naive en la mayoría de las líneas, no en todas |
| Redistribución de demanda entre líneas | Solo descriptivo | H3, n = 1 |
| **Detección temprana de interrupciones** (minutos u horas) | **No** | Export diario; las interrupciones parciales (IN1, IN2) no dejan señal en línea × hora |
| **Localizar estación o tramo** | **No** | No hay datos por estación |
| **Identificar fallas técnicas o hacer mantenimiento predictivo** | **No** | No hay telemetría ni registros de mantenimiento |
| **Recomendar capacidad o frecuencia** | **No** | Se requieren oferta (trenes/h), carga por tramo y matrices O-D; las entradas no son carga |
| **Priorizar inspecciones** | **No** | Se requieren activos, fallas y amenazas georreferenciadas por tramo |

## 2. Fuentes necesarias, en orden de valor esperado

| # | Fuente (dueño probable) | Granularidad mínima | Qué habilita | Cómo se validaría |
|---|---|---|---|---|
| 1 | **Validaciones por estación y torniquete** (Metro, sistema de recaudo) | Estación × 15 min, en tiempo casi real | Detección temprana, localización de la estación afectada, redistribución entre estaciones, problemas de captura por torniquete | Reproducción cronológica con los incidentes de `data/eventos_reales.csv`: recall de IN1–IN3 y tiempo hasta la detección |
| 2 | **Bitácora oficial de incidentes de operación** (Metro, centro de control) | Evento con inicio, fin, línea y tramo | Etiquetas reales para evaluar (hoy hay 3 incidentes verificados), medir falsos positivos y negativos, y aprender firmas | Partición temporal; precisión y recall por tipo de incidente |
| 3 | **Oferta programada y real** (trenes por hora, intervalos, cierres programados) | Línea × hora × día | Separar caída de demanda de reducción de oferta (cierres del 24 y 31 de diciembre, mantenimiento nocturno); recomendaciones de frecuencia | Coherencia entre caídas de demanda y de oferta |
| 4 | **Telemetría de trenes y SCADA** (Metro, mantenimiento) | Segundos o minutos por tren y subsistema | Mantenimiento predictivo, fallas de catenaria o frenos, anticipación técnica | Modelos de supervivencia o de anomalía multivariada con órdenes de trabajo como verdad |
| 5 | **Órdenes de trabajo de mantenimiento** (Metro) | Activo × fecha | Verdad de terreno para el mantenimiento predictivo; priorización de inspecciones | Validación temporal; costo evitado |
| 6 | **Pluviómetros del SIATA** (AMVA) | Estación × 5–15 min | Efecto de la lluvia local por estación. La celda de reanálisis no aportó (H5b) | Repetir H5b con lluvia local y con pronóstico meteorológico emitido el día anterior |
| 7 | **Agenda de eventos masivos** (Alcaldía, escenarios deportivos y culturales) | Evento × fecha × hora × lugar | Explicar picos (T3) y anticipar sobrecarga en estaciones cercanas | Recall de picos explicados |
| 8 | **Matriz Origen-Destino del AMVA** y **población por área de influencia** (DANE / Alcaldía) | Zona | Redistribución entre estaciones y planeación (no operación diaria) | Contraste con la demanda observada por estación |
| 9 | **Geometría oficial de líneas y estaciones** (Metro o datos abiertos) | Polilíneas y puntos con nombre | Mapa geográfico real y cruce con amenazas del POT por tramo (VM_05) | Coincidencia con las páginas oficiales del Metro |

## 3. Requisitos mínimos de calidad para cada nueva fuente
- Marca de tiempo con zona horaria y semántica documentada (inicio o fin de la franja; acumulado o instantáneo).
- Identificadores estables de línea y estación, con tabla de equivalencias y fechas de vigencia.
- Registro de cambios de esquema y de cobertura (estaciones nuevas, cierres).
- Para tiempo real: latencia garantizada y un indicador de dato faltante distinto de cero.
- Licencia o acuerdo de uso que permita el procesamiento.

## 4. Primer paso recomendado
Solicitar al Metro el histórico de **validaciones por estación cada 15 minutos** para 2022–2024 y la **bitácora de incidentes** del mismo periodo. Con esas dos fuentes se puede evaluar, con la misma metodología de reproducción cronológica, si la detección temprana y la localización son viables. Hoy no lo son.
