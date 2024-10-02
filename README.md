# Detección de Anomalías en Afluencia del Metro de Medellín con Autoencoders y Streamlit

Este proyecto tiene como objetivo detectar anomalías en la afluencia de pasajeros del **Metro de Medellín** utilizando **Autoencoders** para el análisis de series temporales y un panel interactivo creado con **Streamlit**. El enfoque se basa en la descomposición de series temporales y el uso de técnicas de aprendizaje profundo para identificar comportamientos anómalos en los datos de afluencia.

## Explicación del aplicativo

El aplicativo web ha sido desarrollado utilizando **Streamlit**, lo que permite una interfaz interactiva y fácil de usar para que el usuario pueda cargar datos, ejecutar predicciones con el modelo de autoencoder y visualizar los resultados de detección de anomalías en tiempo real. El flujo del aplicativo es el siguiente:

1. **Carga de datos**: El usuario puede subir un archivo `.pkl` que contiene los datos de afluencia de pasajeros.
2. **Preprocesamiento**: Los datos cargados son limpiados, normalizados y preparados para su procesamiento posterior.
3. **Detección de anomalías**: Un modelo de autoencoder entrenado previamente se utiliza para analizar los datos y detectar registros que se consideran anómalos. El modelo ha sido entrenado con datos "normales", por lo que cualquier desviación significativa del patrón se considera una anomalía.
4. **Visualización de resultados**: Dependiendo del análisis, el aplicativo mostrará una alerta visual:
   - **Normal**: Se muestra un mensaje de éxito si los datos están dentro de lo esperado.
   - **Anomalía**: Si el modelo detecta una anomalía, se muestra un mensaje de advertencia en rojo, indicando que los datos no siguen los patrones normales.
5. **Gráficas**: La aplicación también genera gráficas que representan la curva de entrenamiento del autoencoder y la proporción de días con outliers (anomalías).

El aplicativo facilita el análisis visual y dinámico de los datos de afluencia de pasajeros, ayudando a los usuarios a identificar comportamientos fuera de lo común que podrían indicar problemas operacionales o cambios inesperados en el flujo de pasajeros.

## Funcionalidades principales

### 1. Carga y limpieza de datos
- Se utiliza una función para cargar los datos desde un archivo de Excel y limpiar las columnas innecesarias, además de formatear las fechas y las horas.

### 2. Filtrado por líneas de servicio
- Filtrado de los datos según las diferentes líneas del servicio del metro, como la **Línea A**.

### 3. Generación de un DataFrame completo
- Generación de un conjunto de datos completo que incluye todas las fechas de un rango dado, con la capacidad de agregar columnas adicionales.

### 4. Cálculo del promedio estacional
- Se calculan promedios estacionales para cada hora del día basado en el comportamiento histórico de los pasajeros.

### 5. Relleno de valores faltantes
- Los valores faltantes se rellenan utilizando los promedios estacionales calculados.

### 6. Eliminación de tendencia y estacionalidad
- Se aplica la descomposición estacional a los datos utilizando métodos aditivos y multiplicativos para eliminar tendencias y estacionalidad.

### 7. Detección de outliers
- Se implementa un método basado en el **IQR (Rango Intercuartílico)** para identificar y marcar outliers en los residuos de la descomposición.

### 8. Visualización de outliers
- Se grafica la proporción de días con al menos un outlier detectado en el conjunto de datos.

### 9. Entrenamiento de Autoencoders
- Entrenamiento de un modelo de **Autoencoder** con **Keras** para detectar patrones en los datos normalizados, con el uso de **Early Stopping** para prevenir el sobreajuste.

### 10. Visualización de la curva de entrenamiento
- Se genera una gráfica de la curva de pérdida durante el entrenamiento y la validación del modelo de autoencoder.

### 11. Evaluación del modelo
- Evaluación del rendimiento del modelo en los datos de test usando curvas ROC y cálculo de errores para identificar el umbral de detección de anomalías.

## Requisitos

- **Python 3.x**
- **Streamlit**
- **scikit-learn**
- **TensorFlow**
- **Pandas**
- **Matplotlib**
- **joblib**
- **etc**

# Cómo usar este proyecto

### 1. Clona el repositorio en tu máquina local:
git clone https://github.com/usuario/AutoencoderMetroAfluencia.git

### 2. Navega al directorio del proyecto:
cd AutoencoderMetroAfluencia

### 3. Instala las dependencias necesarias:
pip install -r requirements.txt

### 4. Ejecuta la aplicación con Streamlit:
streamlit run app.py
