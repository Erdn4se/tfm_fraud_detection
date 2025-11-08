#########################################
# Librerias necesarias para el proyecto #
#########################################

# Analisis y transformacion de datos
import pandas as pd
import numpy as np
# Visualizacion
import matplotlib.pyplot as plt
import seaborn as sns
# Base de datos
import sqlite3
# funciones machine learning
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, classification_report, confusion_matrix
# funciones deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

######################################################
# Funcion para extraer los datos de la base de datos #
######################################################
def extraer_datos_fraude(carpeta_base_datos):
    """ Funcion que realiza la extraccion de datos (SELECT) de todos los campos correspondientes a la tabla de fraude
    Args:
        carpeta_base_datos (str): Carpeta donde se encuentra la base de datos
    Returns:
    pd.DataFrame: DataFrame con los datos de la tabla de fraude
    """
    
    # Abrimos la conexion a la base de datos
    conexion_bbdd = sqlite3.connect(f'{carpeta_base_datos}/base_datos_tfm.db')
    df_select = pd.read_sql('SELECT * FROM tabla_fraude',
                            conexion_bbdd)
    conexion_bbdd.close()
    return df_select


###############################################
# Funcion para hacer SELECT de la BBDD SQLite #
###############################################
def query_bbdd(carpeta_base_datos, nombre_bbdd='base_datos_tfm.db', query='SELECT * FROM tabla_fraude'):
    """ Funcion que realiza la extraccion de datos (SELECT) de todos los campos correspondientes a la tabla de fraude
    Args:
        carpeta_base_datos (str): Carpeta donde se encuentra la base de datos
    Returns:
    pd.DataFrame: DataFrame con los datos de la tabla de fraude
    """
    # Abrimos la conexion a la base de datos
    conexion_bbdd = sqlite3.connect(f'{carpeta_base_datos}/base_datos_tfm.db')
    df_select = pd.read_sql(query,
                            conexion_bbdd)
    conexion_bbdd.close()
    return df_select

#####################################################################
# Funcion para transformar y aplicar split de datos en train y test #
#####################################################################
def preparar_datos(df, n_columnas_x=7, pct_test=0.20, semilla=12345):
    """
    Función que prepara los datos para el entrenamiento y test del modelo.
    argumentos:
    df (DataFrame): El DataFrame que contiene los datos.
    n_columnas_x (int): Número de columnas a usar como características (features).
    semilla (int): Semilla para la reproducibilidad.
    pct_test (float): Porcentaje de datos a usar para el conjunto de prueba.
    return:
    X_train, X_test, y_train, y_test: Conjuntos de entrenamiento y prueba.
    """
    X = df.iloc[:, :n_columnas_x]
    y = df.iloc[:, n_columnas_x]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pct_test, random_state=semilla)
    return X_train, X_test, y_train, y_test

#####################################################
# Funcion para hacer plot de la matriz de confusión #
#####################################################
def plot_confusion_matrix(c_matrix, title='Matriz de Confusión', figsize=(6,4)):
    plt.figure(figsize=figsize)
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Etiquetas Predichas')
    plt.ylabel('Etiquetas Reales')
    plt.title(title)
    plt.show()

######################################################
# Funcion para evaluar el modelo de machine learning #
######################################################
def evaluar_modelo(model, X_test, y_test):
    # Realizamos predicciones
    y_pred = model.predict(X_test)
    # Evaluamos el modelo
    # a. Precisión (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    A. Precisión del modelo: {accuracy:.4f}")
    # b. F1 score
    f1 = f1_score(y_test, y_pred)
    print(f"    B. F1 Score: {f1:.4f}")
    # c. AUC-ROC
    y_proba = model.predict_proba(X_test)[:, 1] 
    auc = roc_auc_score(y_test, y_proba)
    print(f"    C. AUC: {auc:.4f}")
    # d. Recall
    recall_minoritaria = recall_score(y_test, y_pred, pos_label=1) 
    print(f"    D. Recall (Sensibilidad) Clase Minoritaria (1): {recall_minoritaria:.4f}")
    # e. Metricas desagregadas (Reporte de Clasificación)
    print("    E. Metricas desagregadas:")
    metricas_desagregadas = classification_report(y_test, y_pred)
    print(metricas_desagregadas)
    # f. Matriz de confusión
    c_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Etiquetas Predichas')
    plt.ylabel('Etiquetas Reales')
    plt.title('Matriz de Confusión')
    plt.show()
    return accuracy, f1, auc, recall_minoritaria, metricas_desagregadas, c_matrix

#########################################################################
# Funcion para dividir los datos para modelos de deteccion de anomalias #
#########################################################################
def preparar_datos_anomalias(df, pct_test=0.20, semilla=12345):
    """
    Función para preparar y dividir los datos para modelos de detección de anomalías.
    args:
        df: DataFrame con los datos originales
        pct_test: Porcentaje del conjunto de datos normales a usar como test set
        semilla: semilla para la reproducibilidad
    returns:
        X_train_normal: Datos normales para entrenar el modelo (solo clase 0).
        X_test_eval: Datos para la evaluación (normal + anómalo).
        y_test_true: Etiquetas verdaderas para la evaluación.
    """
    # Identificamos caracteristicas (X) y target (y)
    X = df.drop(columns=['Class'])
    y = df['Class'] 
    # Separamos por campo "Clase" (Normal: 0, Anomalia: 1)
    X_normal = X[y == 0]
    y_normal = y[y == 0]
    X_anomalo = X[y == 1]
    y_anomalo = y[y == 1]
    
    # Dividimos los datos normales en entrenamiento y test
    X_train_normal, X_test_normal, _, y_test_normal = train_test_split(X_normal, y_normal, test_size=pct_test, random_state=semilla)
    
    # Creamos el conjunto de test concatenando normales de test y anomalias
    X_test_eval = pd.concat([X_test_normal, X_anomalo])
    y_test_true = pd.concat([y_test_normal, y_anomalo])
    
    # Mezclamos los datos asegurando el orden de las etiquetas
    X_test_eval = X_test_eval.sample(frac=1, random_state=semilla)
    y_test_true = y_test_true.loc[X_test_eval.index] 
    
    return X_train_normal, X_test_eval, y_test_true

################################
# Autoencoder simple (3 capas) #
################################
def autoencoder(input_dim, neuronas_capa1=16, neuronas_capa2=8, neuronas_capa3=4):
    """Autoenconder simple con 3 capas en el encoder y 3 en el decoder
    Args:
        input_dim (int): Dimensión de entrada (número de características)
        neuronas_capa1 (int): Número de neuronas en la primera capa oculta
        neuronas_capa2 (int): Número de neuronas en la segunda capa oculta
        neuronas_capa3 (int): Número de neuronas en la tercera capa oculta
    Returns:
        nn.Module: Modelo de autoencoder
    """
    encoder = nn.Sequential(
        nn.Linear(input_dim, neuronas_capa1),
        nn.ReLU(),
        nn.Linear(neuronas_capa1, neuronas_capa2),
        nn.ReLU(),
        nn.Linear(neuronas_capa2, neuronas_capa3),
        nn.ReLU()
    )
    decoder = nn.Sequential(
        nn.Linear(neuronas_capa3, neuronas_capa2),
        nn.ReLU(),
        nn.Linear(neuronas_capa2, neuronas_capa1),
        nn.ReLU(),
        nn.Linear(neuronas_capa1, input_dim),
        nn.Identity()
    )
    return nn.Sequential(encoder, decoder) 

########################################
# Entrenamiento del Autoencoder simple #
########################################
def entrenar_autoencoder(model, dataloader, epochs=15, learning_rate=0.001):
    """Funcion para entrenar el autoencoder.
    Args:
        model (nn.Module): Modelo de autoencoder
        dataloader (DataLoader): DataLoader con los datos de entrenamiento
        epochs (int): Número de épocas
        learning_rate (float): Ratio de aprendizaje
    Returns:
        list: Historial de la pérdida (MSE promedio) por época.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    # Lista para almacenar la pérdida promedio por época
    historial_perdida = [] 
    
    for epoch in range(epochs):
        total_perdida_epoca = 0
        num_batches = 0
        
        for data in dataloader:
            inputs = data[0]
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_perdida_epoca += loss.item()
            num_batches += 1
            
        # Calcula la pérdida promedio para la época y la registra
        perdida_promedio = total_perdida_epoca / num_batches
        historial_perdida.append(perdida_promedio)
        print(f'Epoca {epoch+1}/{epochs}, Perdida: {perdida_promedio:.6f}')
        
    return historial_perdida

###########################################################
# Calculo del error de reconstruccion (grado de anomalia) #
###########################################################
def calcular_error_reconstruccion(model, X_df):
    """Funcion que calcula el error de reconstrucción (MSE) para cada fila
    Args:
        model (nn.Module): Modelo de autoencoder entrenado
        X_df (pd.DataFrame or np.ndarray): Datos de entrada (normalizados)
    Returns:
        np.ndarray: Array con el error de reconstrucción para cada fila"""
    model.eval()
    # 1. Convertimos a tensor
    X_numpy = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    # 2. Prediccion y calculo de error
    with torch.no_grad():
        reconstruccion = model(X_tensor).numpy()
    # Calculamos el MSE por fila
    # Comparamos el original (X_numpy) con el reconstruido (reconstruccion)
    errors = np.mean(np.square(X_numpy - reconstruccion), axis=1)
    return errors.reshape(-1, 1)


def evaluar_modelo_ajustado(y_test, y_pred_ajustado, y_proba):
    """
    Evalúa el modelo usando las predicciones binarias ya ajustadas (y_pred_ajustado) 
    y las probabilidades (y_proba).
    """
    # a. Precisión (accuracy)
    accuracy = accuracy_score(y_test, y_pred_ajustado)
    print(f"    A. Precisión del modelo: {accuracy:.4f}")
    
    # b. F1 score
    f1 = f1_score(y_test, y_pred_ajustado)
    print(f"    B. F1 Score: {f1:.4f}")
    
    # c. AUC-ROC (usa las probabilidades originales, no afectadas por el umbral)
    auc = roc_auc_score(y_test, y_proba)
    print(f"    C. AUC: {auc:.4f}")
    
    # d. Recall
    recall_minoritaria = recall_score(y_test, y_pred_ajustado, pos_label=1) 
    print(f"    D. Recall (Sensibilidad) Clase Minoritaria (1): {recall_minoritaria:.4f} (Umbral Ajustado)")
    
    # e. Metricas desagregadas (Reporte de Clasificación)
    print("    E. Metricas desagregadas:")
    metricas_desagregadas = classification_report(y_test, y_pred_ajustado)
    print(metricas_desagregadas)
    
    # f. Matriz de confusión
    c_matrix = confusion_matrix(y_test, y_pred_ajustado)
    plt.figure(figsize=(6,4))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Etiquetas Predichas')
    plt.ylabel('Etiquetas Reales')
    plt.title(f'Matriz de Confusión (Umbral Ajustado)')
    plt.show()
    
    return accuracy, f1, auc, recall_minoritaria, metricas_desagregadas, c_matrix