# Protocolo para validar la relación entre anomalías detectadas e incidentes reales

**Problema:** sin un registro independiente de incidentes no se puede demostrar que una alerta corresponda a un evento operacional verdadero (`docs/validacion/F8_F9_produccion_y_ux.md` §3). Este protocolo define cómo establecer, o refutar, esa relación con datos operacionales y revisión humana **antes** de usar el sistema para decisiones.

## 1. Fuentes de verdad necesarias (a solicitar al Metro de Medellín)
| Fuente | Contenido mínimo | Uso |
|---|---|---|
| Bitácora del Centro de Control | inicio y fin (minuto), línea, tramo/estaciones, tipo (falla técnica, persona en la vía, suspensión, operación parcial, evacuación), afectación | verdad de terreno de incidentes |
| Registro de no cobro, gratuidades y horarios especiales | fecha, horas, líneas | separar efectos de medición |
| Agenda de eventos masivos y cierres programados | fecha, horas, estaciones | separar eventos conocidos |
| Datos de validación por estación y en intervalos de 15 min (y salidas, si existen) | conteos por estación | localización y sensibilidad a interrupciones parciales |
| Registro de calidad del sistema de recaudo | caídas de torniquetes y envíos tardíos | separar fallas de captura |

## 2. Diseño del estudio
1. **Modo sombra (*shadow mode*) prospectivo de 6 meses.** El sistema corre a diario (`scripts/run_daily.py`), pero sus alertas no se muestran a la operación. Se congelan modelo y umbral (hash del artefacto en cada registro).
2. **Revisión ciega por pares.** Dos analistas, que no ven el score, clasifican cada día del periodo usando solo la bitácora y las fuentes de §1:
   - incidente que afecta la Línea A (total / parcial);
   - evento conocido (calendario, gratuidad, evento masivo);
   - problema de datos;
   - sin novedad.

   Se resuelven los desacuerdos y se mide el acuerdo (κ de Cohen).
3. **Cruce con las alertas** (después de cerrar las etiquetas) y cálculo de:
   - sensibilidad por tipo de incidente (total o parcial; hora pico o valle; duración);
   - precisión (valor predictivo positivo) de "anomalía estadística" para "incidente";
   - tasa de alertas sin explicación;
   - retardo entre el inicio del incidente (bitácora) y el dato que dispara la alerta;
   - proporción de incidentes que el Centro de Control ya conocía antes de la alerta (valor incremental).
4. **Tamaño de muestra.** Con ~2 % de días con alerta (~4/mes), 6 meses dan ~24 alertas. Para estimar una precisión de 0,5 con ±0,2 (IC 95 %) se necesitan ≥ 24 alertas. Para la sensibilidad, que dependerá del número de incidentes registrados, se informa el IC exacto de Clopper-Pearson.
5. **Criterios de aceptación** (a acordar con el Metro antes de empezar). Ejemplo:
   - sensibilidad ≥ 0,8 en suspensiones totales de ≥ 30 min en hora pico;
   - precisión ≥ 0,5 para "incidente o problema de datos";
   - valor incremental > 0: al menos una fracción de las alertas aporta información que la operación no tenía.

   Si no se cumplen, el sistema queda limitado a control de calidad de datos y revisión retrospectiva.

## 3. Revisión humana de cada alerta en operación (flujo propuesto)
1. **Anomalía detectada** (lo que entrega el sistema): un score sobre el umbral, con horas y magnitud.
2. **Sospecha operacional** (decisión del analista): el analista cruza la alerta con la bitácora, el calendario, la gratuidad y la calidad de datos, y la marca como "sospecha operacional", "explicada por calendario o medición", "problema de datos" o "sin explicación".
3. **Incidente confirmado** (solo con evidencia operacional): requiere un registro en la bitácora del Centro de Control. El sistema nunca lo declara por sí solo.
4. Cada decisión humana se guarda junto al registro de auditoría (fecha, categoría del sistema, clasificación humana, fuente) para reentrenar y recalibrar con verdad de terreno.

## 4. Lo que este protocolo no puede corregir
Si la información operativa relevante no altera las entradas de la línea (interrupciones parciales, desvíos internos), ningún ajuste del modelo lo hará visible: hace falta un dato más fino (estación, 15 min, salidas o posición de trenes).
