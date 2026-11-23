# App legacy (referencia histórica)

`app_legacy.py` y `utils.py` son la aplicación original (autoencoder + umbral fijo de 827 pasajeros). Se conservan
solo como referencia de la auditoría (`audit/AUDITORIA.md`); **no deben usarse para decisiones**:
- el modelo se entrenó con fuga de datos (C2) y con etiquetas defectuosas (C1, C5);
- con el protocolo correcto queda por debajo del modelo actual en todos los criterios (`docs/validacion/F4.md`).

La única modificación posterior a la auditoría es la de seguridad (F1): deserialización restringida y validación.
Ejecución (requiere `requirements-research.txt`): `streamlit run legacy/app_legacy.py` desde la raíz del repositorio.
