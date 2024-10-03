import streamlit as st
from utils import *

# Características básicas de la página
st.set_page_config(page_icon="🚆", page_title="Detección de Anomalías en Afluencia del Metro de Medellín", layout="wide")

# Crear una fila para centrar el título y el logo
c1, c2, c3 = st.columns([1, 6, 1])

# Centrar el título y el logo en la segunda columna
with c2:
    # Crear una fila dividida en dos partes: título y logo
    col1, col2 = st.columns([8, 2])  # La primera columna más grande para el título, la segunda más pequeña para el logo

    # Colocar el título en la primera columna
    with col1:
        st.title("Detección de Anomalías en Afluencia del Metro de Medellín con Autoencoders")

    # Colocar el logo en la segunda columna, proporcional al tamaño del texto
    with col2:
        st.image("./assets/logo.png", width=200)  # Ajusta el tamaño del logo para que sea proporcional al título

# Definir las columnas para el cargador de archivos
c29, c30, c31 = st.columns([1, 6, 1])  # 3 columnas: 10%, 60%, 10%

UMBRAL = 1680  # Umbral ajustable para la predicción

with c30:
    uploaded_file = st.file_uploader(
        "", type='pkl',
        key="1",
    )

    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .pkl que acaba de subir")

        # Espacio para el mensaje de estado
        info_box = st.empty()

        # Mostrar el mensaje de espera
        info_box.info("🕒 Realizando la clasificación...")

        # Cargar el dato desde el archivo .pkl
        dato = leer_dato(uploaded_file)

        # Cargar el modelo preentrenado en formato .keras o .h5
        autoencoder, scaler = cargar_modelo_preentrenado('./dev/results/tf_modelo_77_30_38.keras', './dev/results/scaler_77_30_38.pkl')

        # Realizar la predicción
        prediccion = predecir(autoencoder, dato, UMBRAL, scaler)
        categoria = obtener_categoria(prediccion)

        # Limpiar el mensaje de espera y mostrar el resultado de forma llamativa
        info_box.empty()

        # Mostrar el resultado, usando rojo si es anormal y verde si es normal
        if categoria == "Anormal":
            st.error(f"⚠️ El dato analizado corresponde a un sujeto: **{categoria}**")
        else:
            st.success(f"🎉 El dato analizado corresponde a un sujeto: **{categoria}**")

    else:
        st.info(
            "👆 Debe cargar primero un dato con extensión .pkl"
        )
        st.stop()