import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from collections import deque
import sys

from statsmodels.tsa.seasonal import seasonal_decompose, STL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

from sklearn.metrics import (
                            roc_curve,
                            auc, accuracy_score,
                            precision_score, 
                            recall_score,
                            f1_score
                            )
import tensorflow as tf
from pickle import dump

# !{sys.executable} -m pip install pyyaml
import yaml

# !{sys.executable} -m pip torch torchaudio torchvision
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())  # Verifica si se reconoce la GPU

# !{sys.executable} -m pip streamlit
# import streamlit
# 1. Función para cargar y limpiar los datos
def cargar_y_limpiar_datos(file_path):
    data = pd.read_excel(file_path, header=0)
    data = data.drop(['Total general (Número de pasajeros)'], axis=1)
    data['Día'] = pd.to_datetime(data['Día'], dayfirst=True)
    data.columns = [str(col) if isinstance(col, datetime.time) else col for col in data.columns]
    horas = [col for col in data.columns if ':' in col]
    return data, horas

# 2. Función para filtrar datos de una línea específica
def filtrar_por_linea(data, linea_servicio='línea a'):
    return data[data['Línea de Servicio'].str.lower() == linea_servicio]

# 3. Función para generar el DataFrame completo de fechas y agregar columnas adicionales
def generar_dataframe_completo(data_linea_a, start, end):
    fechas_2023 = pd.DataFrame({'Día': pd.date_range(start=start, end=end)})
    data_linea_a_full = pd.merge(fechas_2023, data_linea_a, on='Día', how='left')
    data_linea_a_full['Día de la Semana'] = data_linea_a_full['Día'].dt.day_name()
    data_linea_a_full.columns = [str(col) if isinstance(col, datetime.time) else col for col in data_linea_a_full.columns]
    return data_linea_a_full

# 4. Función para calcular el promedio estacional por día de la semana
def calcular_promedio_estacional(data, horas):
    return data.groupby('Día de la Semana')[horas].median().round(0)

# 5. Función para rellenar los valores faltantes
def rellenar_faltantes(data, promedio_estacional, horas):
    # Asegúrate de que 'Día' esté presente en el DataFrame
    if 'Día de la Semana' not in data.columns:
        raise ValueError("El DataFrame 'data' debe contener una columna llamada 'Día' que indique el día de la semana.")

    # Itera sobre las horas para rellenar los valores faltantes
    for hora in horas:
        if hora in data.columns:
            # Rellena los valores faltantes según el promedio estacional
            data[hora] = data.apply(
                lambda row: promedio_estacional.loc[row['Día de la Semana'], hora] if pd.isnull(row[hora]) else row[hora], axis=1
            )
        else:
            print(f"La columna {hora} no existe en el DataFrame, omitiendo...")
        
    data['Línea de Servicio'] = data['Línea de Servicio'].fillna(method='ffill') 

    return data

# 6. Función para eliminar tendencia y estacionalidad con STL incluido
# def eliminar_tendencia_estacionalidad(data, horas):
#     descomposiciones = {}
#     for hora in horas:
#         serie = data.set_index('Día')[hora].dropna()
#         mse_dict = {}
        
#         # Descomposición aditiva
#         try:
#             descomp_aditiva = seasonal_decompose(serie, model='additive', period=7)
#             residuo_aditivo = descomp_aditiva.resid.dropna()
#             mse_aditivo = np.mean(np.square(residuo_aditivo))
#             mse_dict['aditiva'] = mse_aditivo
#         except Exception as e:
#             print(f"Error en descomposición aditiva para {hora}: {e}")
#             mse_dict['aditiva'] = np.inf
        
#         # Descomposición multiplicativa
#         try:
#             descomp_multiplicativa = seasonal_decompose(serie, model='multiplicative', period=7)
#             residuo_multiplicativa = descomp_multiplicativa.resid.dropna()
#             mse_multiplicativa = np.mean(np.square(residuo_multiplicativa))
#             mse_dict['multiplicativa'] = mse_multiplicativa
#         except Exception as e:
#             print(f"Error en descomposición multiplicativa para {hora}: {e}")
#             mse_dict['multiplicativa'] = np.inf
        
#         # Descomposición STL
#         try:
#             stl = STL(serie, period=7)
#             descomp_stl = stl.fit()
#             residuo_stl = descomp_stl.resid.dropna()
#             mse_stl = np.mean(np.square(residuo_stl))
#             mse_dict['STL'] = mse_stl
#         except Exception as e:
#             print(f"Error en descomposición STL para {hora}: {e}")
#             mse_dict['STL'] = np.inf
        
#         # Selección del método con menor error
#         metodo_min_error = min(mse_dict, key=mse_dict.get)
#         if metodo_min_error == 'aditiva':
#             print(f"Usando descomposición aditiva para {hora}")
#             descomposiciones[hora] = descomp_aditiva
#         elif metodo_min_error == 'multiplicativa':
#             print(f"Usando descomposición multiplicativa para {hora}")
#             descomposiciones[hora] = descomp_multiplicativa
#         elif metodo_min_error == 'STL':
#             print(f"Usando descomposición STL para {hora}")
#             descomposiciones[hora] = descomp_stl
#     return descomposiciones

# 6. Función para eliminar tendencia y estacionalidad con STL incluido
def eliminar_tendencia_estacionalidad(data, horas):
    descomposiciones = {}
    data = data.set_index('Día')  # Establecer el índice fuera del bucle
    
    for hora in horas:
        if hora not in data.columns:
            print(f"La columna {hora} no está presente en los datos.")
            continue
        
        serie = data[hora].dropna()
        
        # Verificar si la serie está vacía
        if serie.empty:
            print(f"La serie temporal para {hora} está vacía después de eliminar valores nulos.")
            continue
        
        mse_dict = {}
        
        # Descomposición aditiva
        try:
            descomp_aditiva = seasonal_decompose(serie, model='additive', period=7)
            residuo_aditivo = descomp_aditiva.resid.dropna()
            mse_aditivo = np.mean(np.square(residuo_aditivo))
            mse_dict['aditiva'] = mse_aditivo
        except ValueError as e:
            print(f"Error de valor en descomposición aditiva para {hora}: {e}")
            mse_dict['aditiva'] = np.inf
        except Exception as e:
            print(f"Error inesperado en descomposición aditiva para {hora}: {e}")
            mse_dict['aditiva'] = np.inf
        
        # Descomposición multiplicativa
        try:
            descomp_multiplicativa = seasonal_decompose(serie, model='multiplicative', period=7)
            residuo_multiplicativa = descomp_multiplicativa.resid.dropna()
            mse_multiplicativa = np.mean(np.square(residuo_multiplicativa))
            mse_dict['multiplicativa'] = mse_multiplicativa
        except ValueError as e:
            print(f"Error de valor en descomposición multiplicativa para {hora}: {e}")
            mse_dict['multiplicativa'] = np.inf
        except Exception as e:
            print(f"Error inesperado en descomposición multiplicativa para {hora}: {e}")
            mse_dict['multiplicativa'] = np.inf
        
        # Descomposición STL
        try:
            stl = STL(serie, period=7)
            descomp_stl = stl.fit()
            residuo_stl = descomp_stl.resid.dropna()
            mse_stl = np.mean(np.square(residuo_stl))
            mse_dict['STL'] = mse_stl
        except ValueError as e:
            print(f"Error de valor en descomposición STL para {hora}: {e}")
            mse_dict['STL'] = np.inf
        except Exception as e:
            print(f"Error inesperado en descomposición STL para {hora}: {e}")
            mse_dict['STL'] = np.inf
        
        # Selección del método con menor error
        if all(mse == np.inf for mse in mse_dict.values()):
            print(f"No se pudo realizar ninguna descomposición válida para {hora}.")
            continue
        
        metodo_min_error = min(mse_dict, key=mse_dict.get)
        if metodo_min_error == 'aditiva':
            print(f"Usando descomposición aditiva para {hora}")
            descomposiciones[hora] = descomp_aditiva
        elif metodo_min_error == 'multiplicativa':
            print(f"Usando descomposición multiplicativa para {hora}")
            descomposiciones[hora] = descomp_multiplicativa
        elif metodo_min_error == 'STL':
            print(f"Usando descomposición STL para {hora}")
            descomposiciones[hora] = descomp_stl

    return descomposiciones

# 7. Función para detectar outliers usando el método IQR
def detectar_outliers_en_residuos(descomposiciones, umbral=1.5, data=None):
    outliers_matrix = pd.DataFrame(0, index=descomposiciones[list(descomposiciones.keys())[0]].resid.dropna().index, columns=descomposiciones.keys())
    for hora, descomposicion in descomposiciones.items():
        if hora != list(descomposiciones.keys())[-1]:
            residuo = pd.DataFrame(descomposicion.resid.dropna()).merge(data, on='Día', how='left')
            residuo.index = outliers_matrix.index
            for day in data['Día de la Semana'].unique():
                residuo_estacional = residuo[residuo['Día de la Semana'] == day][hora]
                Q1 = residuo_estacional.quantile(0.25)
                Q3 = residuo_estacional.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - umbral * IQR
                upper_bound = Q3 + umbral * IQR
                outliers = ((residuo_estacional < lower_bound) | (residuo_estacional > upper_bound)).astype(int)
                outliers_matrix.loc[outliers.index, hora] = outliers

    # Columna que cuenta el número total de outliers por día
    outliers_matrix['Total_Outliers'] = outliers_matrix.sum(axis=1)
    
    # Columna que indica si existe al menos un outlier en alguna de las horas (1 si al menos uno, 0 si no) -> si hay mas de 1 (¿Una linea?, ¿Una hora influye en las demas?)
    outliers_matrix['Any_Outlier'] = (outliers_matrix['Total_Outliers'] > 0).astype(int)        
    
    return outliers_matrix

# 8. Función para graficar la proporción de días con outliers
def graficar_proporcion_outliers(data):
    proporcion_outliers = data['Hay_Outlier'].value_counts(normalize=True)
    plt.figure(figsize=(6,4))
    proporcion_outliers.plot(kind='bar', color=['blue', 'orange'], grid=True)
    plt.title('Proporción de días con al menos un outlier')
    plt.xlabel('Hay Outlier')
    plt.ylabel('Proporción')
    plt.xticks(ticks=[0, 1], labels=['No hay outliers', 'Hay al menos un outlier'], rotation=0)
    plt.ylim(0, 1)
    plt.show()

# Función para entrenar un autoencoder con mayor capacidad (32 -> 8 neuronas)
def entrenar_autoencoder(train_data_normalized, test_data_normalized, input_dim):
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    
    # Encoder
    encoded = tf.keras.layers.Dense(16, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(4, activation='relu')(encoded)
    
    # Decoder
    decoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(16, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Construimos el modelo
    autoencoder = tf.keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mae')

    # Implementamos Early Stopping para prevenir sobreajuste
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entrenamiento
    history = autoencoder.fit(train_data_normalized, train_data_normalized,
                              epochs=500, batch_size=32,
                              validation_data=(test_data_normalized, test_data_normalized),
                              callbacks=[early_stopping],
                              verbose=1)
    
    return autoencoder, history

# # Función para entrenar un autoencoder con mayor capacidad (32 -> 8 neuronas)
# class Autoencoder(torch.nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()

#         #Codificador
#         self.cod1 = torch.nn.Linear(in_features=20, out_features=16) # Entrada: 20 muestras
#         self.cod2 = torch.nn.Linear(in_features=16, out_features=8)
#         self.cod3 = torch.nn.Linear(in_features=8, out_features=4)
        
#         #Decodificador
#         self.dec1 = torch.nn.Linear(in_features=4, out_features=8)
#         self.dec2 = torch.nn.Linear(in_features=8, out_features=16)
#         self.dec3 = torch.nn.Linear(in_features=16, out_features=20) # Salida: 20 muestras
        
#     def forward(self, x):
#         x = torch.nn.functional.relu(self.cod1(x))
#         x = torch.nn.functional.relu(self.cod2(x))
#         x = torch.nn.functional.relu(self.cod3(x))
        
#         x = torch.nn.functional.relu(self.dec1(x))
#         x = torch.nn.functional.relu(self.dec2(x))
#         x = torch.sigmoid(self.dec3(x))
        
#         return x

# 10. Función para graficar la curva de entrenamiento
def graficar_curva_entrenamiento(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.title('Curva de Error durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MAE)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./results/train_curve.png')
    plt.show()

# 11. Función para evaluar el modelo con curva ROC
def evaluar_autoencoder(autoencoder, test_data_normalized, test_data, scaler, outlier_labels, umbral):
    predicted_normalized = autoencoder.predict(test_data_normalized)
    predicted = scaler.inverse_transform(predicted_normalized)
    errors = np.mean(np.abs(test_data - predicted), axis=1)

    errors_normales = errors[outlier_labels == 0]
    threshold = errors_normales.mean() + umbral * errors_normales.std()

    return threshold, errors

def graficar_casos_outliers_vs_normales(autoencoder, scaler, data_linea_a_full_outliers, horas_outliers_columns, n=11):
    # Selección de un caso sin outliers y un caso con outliers
    sin_outliers = data_linea_a_full_outliers[data_linea_a_full_outliers['Any_Outlier_outlier'] == 0].iloc[n]
    con_outliers = data_linea_a_full_outliers[data_linea_a_full_outliers['Any_Outlier_outlier'] == 1].iloc[n]

    # Convertir las columnas de las horas de datos reales y reconstruidos a arrays
    # Datos reales y reconstruidos para un caso sin outliers
    real_sin_outliers = sin_outliers[horas_outliers_columns].values
    reconstruido_sin_outliers = scaler.inverse_transform(autoencoder.predict(scaler.transform([real_sin_outliers])))[0]

    # Datos reales y reconstruidos para un caso con outliers
    real_con_outliers = con_outliers[horas_outliers_columns].values
    reconstruido_con_outliers = scaler.inverse_transform(autoencoder.predict(scaler.transform([real_con_outliers])))[0]

    # Identificar las horas que tienen outliers en el caso con outliers
    horas_con_outliers = [col for col in horas_outliers_columns if con_outliers[f"{col}_outlier"] == 1]

    # 2. Creación de los subplots
    plt.figure(figsize=(14, 6))

    # 2.1 Subplot para caso sin outliers
    plt.subplot(1, 2, 1)
    plt.plot(horas_outliers_columns, real_sin_outliers, label='Real', color='blue', marker='o')
    plt.plot(horas_outliers_columns, reconstruido_sin_outliers, label='Reconstruido', color='green', marker='x')
    plt.title('Caso sin Outliers')
    plt.xlabel('Hora')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotar los xticks para que se vean mejor
    plt.ylabel('Afluencia de Pasajeros')
    plt.legend()

    # 2.2 Subplot para caso con outliers
    plt.subplot(1, 2, 2)
    plt.plot(horas_outliers_columns, real_con_outliers, label='Real', color='blue', marker='o')
    plt.plot(horas_outliers_columns, reconstruido_con_outliers, label='Reconstruido', color='green', marker='x')

    # Marcar las horas con outliers en rojo
    for hora_outlier in horas_con_outliers:
        index = horas_outliers_columns.index(hora_outlier)
        plt.scatter(hora_outlier, real_con_outliers[index], color='red', label='Outlier' if hora_outlier == horas_con_outliers[0] else "", zorder=5)

    plt.title('Caso con Outliers')
    plt.xlabel('Hora')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotar los xticks para que se vean mejor
    plt.ylabel('Afluencia de Pasajeros')
    plt.legend()

    # Mostrar los subplots
    plt.tight_layout()
    plt.show()