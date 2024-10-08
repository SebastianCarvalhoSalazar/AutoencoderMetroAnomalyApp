import pickle
import numpy as np
from tensorflow import keras
import joblib
import sklearn

def leer_dato(uploaded_file):
    dato = pickle.loads(uploaded_file.getvalue())
    print(dato)
    return dato

def cargar_modelo_preentrenado(model_path, scaler_path):
    # Cargar el modelo preentrenado de TensorFlow/Keras
    modelo = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return modelo, scaler

def predecir(modelo, datos, umbral, scaler):
    # Modificamos para usar TensorFlow/Keras
    # Realizamos la predicción con el modelo de Keras
    reconstrucciones = modelo.predict(np.array([datos]))  # Predecimos con un solo dato

    scaled_reconstrucciones = scaler.inverse_transform(reconstrucciones.reshape(1, -1))
    scaled_datos = scaler.inverse_transform(datos.reshape(1, -1))

    # Calculamos la pérdida utilizando MAE
    perdida = np.mean(np.abs(scaled_reconstrucciones - scaled_datos))
    print(f"PERDIDA: {np.round(perdida,0)}")

    return perdida < umbral  # Comparamos con el umbral

def obtener_categoria(comparaciones):
    # No se requiere el uso de "for" pues tendremos sólo 1 dato
    if comparaciones:
        categoria = 'Normal'
    else:
        categoria = 'Anormal'

    return categoria