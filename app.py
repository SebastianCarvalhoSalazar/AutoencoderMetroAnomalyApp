import streamlit as st
import numpy as np
from utils import *
import matplotlib.pyplot as plt

# Función para codificar horas en seno y coseno
def codificar_horas_en_seno_cos(hora_inicial, hora_final):
    # Crear un vector de horas entre 4 y 23 (incluyendo 23)
    horas = np.arange(4, 24)  # Horas entre 4AM y 11PM (23 inclusive)
    
    # Reordenar el vector de horas para que comience desde la hora_inicial hasta la hora_final
    if hora_inicial <= hora_final:
        horas_ordenadas = np.concatenate((horas[hora_inicial-4:hora_final-3], horas[:hora_inicial-4]))
    else:
        # Si la hora inicial es mayor que la hora final, entonces necesitamos envolver
        horas_ordenadas = np.concatenate((horas[hora_inicial-4:], horas[:hora_final-3]))
    
    print("Horas:", horas_ordenadas, "-->", len(horas_ordenadas))
    # Codificar las horas ordenadas en dominio seno y coseno
    horas_seno = np.sin(2 * np.pi * horas_ordenadas / 24)
    horas_coseno = np.cos(2 * np.pi * horas_ordenadas / 24)

    return horas_seno, horas_coseno

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

UMBRAL = 7902  # Umbral ajustable para la predicción

with c30:
    # Casillas para seleccionar la hora inicial y la hora final
    hora_inicial = st.number_input("Seleccione la hora inicial (4-23)", min_value=4, max_value=23, value=4)
    hora_final = st.number_input("Seleccione la hora final (4-23)", min_value=4, max_value=23, value=23)

    uploaded_file = st.file_uploader("", type='pkl', key="1")

    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .pkl que acaba de subir")

        # Espacio para el mensaje de estado
        info_box = st.empty()
        info_box.info("🕒 Realizando la clasificación...")

        # Cargar el dato desde el archivo .pkl
        dato_afluencia = leer_dato(uploaded_file)

        # Generar los vectores de horas codificadas en seno y coseno
        horas_seno, horas_coseno = codificar_horas_en_seno_cos(hora_inicial, hora_final)

        # Cargar el modelo preentrenado en formato .keras o .h5
        autoencoder, scaler = cargar_modelo_preentrenado('./dev/results/tf_modelo_52_28_37.keras', './dev/results/scaler_52_28_37.pkl')

        # Escalar los datos de afluencia antes de pasarlos al modelo
        # dato_afluencia_escalado = scaler.transform(dato_afluencia.reshape(1, -1))

        # Realizar la predicción pasando afluencia, seno y coseno como inputs separados
        reconstrucciones = autoencoder.predict([
            dato_afluencia.reshape(1, -1),                 # Afluencia escalada
            horas_seno.reshape(1, -1),                     # Horas codificadas en seno
            horas_coseno.reshape(1, -1)                    # Horas codificadas en coseno
        ])  # Predicción completa
        
        scaled_reconstrucciones = scaler.inverse_transform(reconstrucciones.reshape(1, -1))
        scaled_datos = scaler.inverse_transform(dato_afluencia.reshape(1, -1))

        # Calcular la pérdida
        prediccion = predecir(
            autoencoder,
            dato_afluencia.reshape(1, -1),  # Afluencia escalada
            horas_seno.reshape(1, -1),      # Horas codificadas en seno
            horas_coseno.reshape(1, -1),    # Horas codificadas en coseno
            UMBRAL,
            scaler
        )
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