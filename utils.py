import pickle
import numpy as np
from tensorflow import keras
import joblib
import sklearn

def leer_dato(uploaded_file):
    dato = pickle.loads(uploaded_file.getvalue())
    return dato

def cargar_modelo_preentrenado(model_path, scaler_path):
    # Cargar el modelo preentrenado de TensorFlow/Keras
    modelo = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return modelo, scaler

def predecir(modelo, datos_afluencia, datos_horas_seno, datos_horas_coseno, umbral, scaler):
    # Predicción usando las tres entradas: afluencia, horas en seno y horas en coseno
    reconstrucciones = modelo.predict([datos_afluencia, datos_horas_seno, datos_horas_coseno])
    
    # Invertir la normalización de los datos y las reconstrucciones
    scaled_reconstrucciones = scaler.inverse_transform(reconstrucciones.reshape(1, -1))
    scaled_datos = scaler.inverse_transform(datos_afluencia.reshape(1, -1))
    
    # Calcular la pérdida utilizando MAE
    perdida = np.mean(np.abs(scaled_reconstrucciones - scaled_datos))
    print(f"PERDIDA: {np.round(perdida, 0)}")

    # Comparar con el umbral
    return perdida < umbral

def obtener_categoria(comparaciones):
    # No se requiere el uso de "for" pues tendremos sólo 1 dato
    if comparaciones:
        categoria = 'Normal'
    else:
        categoria = 'Anormal'

    return categoria