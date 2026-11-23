import io
import pickle

import joblib
import numpy as np

N_HORAS = 20

# Solo se permiten las clases que numpy usa para serializar un ndarray.
# Cualquier otra referencia global (os.system, builtins.eval, ...) se rechaza: evita la
# ejecución remota de código al cargar archivos subidos por el usuario (auditoría C7).
_PERMITIDOS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _UnpicklerRestringido(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _PERMITIDOS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Referencia no permitida en el archivo: {module}.{name}")


def leer_dato(uploaded_file):
    """Carga un vector escalado de 20 horas desde un .pkl de forma segura y lo valida."""
    try:
        dato = _UnpicklerRestringido(io.BytesIO(uploaded_file.getvalue())).load()
    except Exception as e:  # cualquier fallo de deserialización es un archivo inválido
        raise ValueError(f"Archivo .pkl inválido o no permitido: {e}") from None
    if not isinstance(dato, np.ndarray) or not np.issubdtype(dato.dtype, np.number):
        raise ValueError("El archivo debe contener un arreglo numérico de numpy.")
    if dato.shape != (N_HORAS,):
        raise ValueError(f"Se esperaban {N_HORAS} valores (04:00–23:00); se recibió la forma {dato.shape}.")
    if not np.isfinite(dato).all():
        raise ValueError("El arreglo contiene valores faltantes (NaN) o infinitos.")
    return dato.astype(float)


def cargar_modelo_preentrenado(model_path, scaler_path):
    # Importación diferida: TensorFlow solo es necesario para la app legacy.
    from tensorflow import keras

    modelo = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return modelo, scaler


def predecir(modelo, datos, umbral, scaler):
    # Realizamos la predicción con el modelo de Keras
    reconstrucciones = modelo.predict(np.array([datos]), verbose=0)

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
