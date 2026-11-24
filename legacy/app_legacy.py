import streamlit as st
from utils import *
import matplotlib.pyplot as plt

# Características básicas de la página
st.set_page_config(page_icon="🚆", page_title="Detección de Anomalías en Afluencia del Metro de Medellín", layout="wide")

# Crear una fila para centrar el título y el logo
c1, c2, c3 = st.columns([1, 6, 1])

# Centrar el título y el logo en la segunda columna
with c2:
    col1, col2 = st.columns([8, 2])  # La primera columna más grande para el título, la segunda más pequeña para el logo
    with col1:
        st.title("Detección de Anomalías en Afluencia del Metro de Medellín con Autoencoders")
    with col2:
        st.image("./assets/logo.png", width=200)

# Definir las columnas para el cargador de archivos
c29, c30, c31 = st.columns([1, 6, 1])

UMBRAL = 827  # Umbral ajustable para la predicción

with c30:
    uploaded_file = st.file_uploader("", type='pkl', key="1")

    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .pkl que acaba de subir")

        # Espacio para el mensaje de estado
        info_box = st.empty()
        info_box.info("🕒 Realizando la clasificación...")

        # Cargar el dato desde el archivo .pkl
        try:
            dato = leer_dato(uploaded_file)
        except ValueError as e:
            info_box.empty()
            st.error(f"❌ {e}")
            st.stop()

        # Cargar el modelo preentrenado en formato .keras o .h5
        autoencoder, scaler = cargar_modelo_preentrenado('./dev/results/tf_modelo_77_81_78.keras', './dev/results/scaler_77_81_78.pkl')

        # Realizar la predicción
        reconstrucciones = autoencoder.predict(np.array([dato]))  # Predicción completa
        scaled_reconstrucciones = scaler.inverse_transform(reconstrucciones.reshape(1, -1))
        scaled_datos = scaler.inverse_transform(dato.reshape(1, -1))

        # Calcular la pérdida
        prediccion = predecir(autoencoder, dato, UMBRAL, scaler)
        categoria = obtener_categoria(prediccion)

        # Limpiar el mensaje de espera y mostrar el resultado de forma llamativa
        info_box.empty()

        # Mostrar el resultado, usando rojo si es anormal y verde si es normal
        if categoria == "Anormal":
            st.error(f"⚠️ El dato analizado corresponde a un sujeto: **{categoria}**")
        else:
            st.success(f"🎉 El dato analizado corresponde a un sujeto: **{categoria}**")

        # Graficar los datos originales y reconstruidos
        st.subheader("Comparación entre los datos originales y reconstruidos")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(scaled_datos.flatten(), label='Datos Originales', color='blue', lw=2)
        ax.plot(scaled_reconstrucciones.flatten(), label='Reconstrucción del Autoencoder', color='red', lw=2, linestyle='--')
        ax.set_title('Datos Originales vs Reconstrucción')
        ax.set_xlabel('Índice')
        ax.set_ylabel('Afluencia')
        plt.grid(True)
        ax.legend()

        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)

    else:
        st.info("👆 Debe cargar primero un dato con extensión .pkl")
        st.stop()
