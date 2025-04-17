"""
Módulo para cargar y preprocesar los datos de tweets
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .utils import clean_text, apply_oversampling, apply_smote

def load_data(url=None, balance_method=None, random_state=42):
    """
    Carga el dataset de tweets y opcionalmente aplica balanceo de clases
    
    Args:
        url (str): URL del dataset (si es None, usa la URL por defecto)
        balance_method (str): Método de balanceo ('oversampling', 'smote' o None)
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        pandas.DataFrame: DataFrame con los tweets y etiquetas
    """
    if url is None:
        url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    
    # Descargar y cargar el dataset
    csv_path = tf.keras.utils.get_file("twitter_sentiment.csv", url)
    df = pd.read_csv(csv_path)
    
    # Verificar columnas y renombrar si es necesario
    if 'tweet' not in df.columns and 'text' in df.columns:
        df = df.rename(columns={'text': 'tweet'})
    
    # Asegurarse de que tenemos las columnas esperadas
    expected_columns = ['tweet', 'label']
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no está presente en el dataset")
    
    # Limpiar tweets
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)
    
    # Aplicar balanceo de clases si se solicita
    if balance_method == 'oversampling':
        df = apply_oversampling(df, random_state=random_state)
    
    return df

def prepare_data(df, max_words=10000, max_len=100, test_size=0.2, balance_method=None, random_state=42):
    """
    Prepara los datos para entrenamiento, incluyendo tokenización y padding
    
    Args:
        df (pandas.DataFrame): DataFrame con tweets limpios
        max_words (int): Tamaño máximo del vocabulario
        max_len (int): Longitud máxima de cada secuencia
        test_size (float): Proporción de datos para testing
        balance_method (str): Método de balanceo ('smote' o None)
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, tokenizer)
    """
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_tweet'], 
        df['label'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']  # Mantener proporción de clases
    )
    
    # Tokenización
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    # Convertir textos a secuencias
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Padding
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Aplicar SMOTE si se solicita (solo a los datos de entrenamiento)
    if balance_method == 'smote':
        X_train_pad, y_train = apply_smote(X_train_pad, y_train, random_state=random_state)
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer

def get_class_weights(y_train):
    """
    Calcula los pesos de clase para manejar el desbalanceo
    
    Args:
        y_train: Etiquetas de entrenamiento
        
    Returns:
        dict: Diccionario con los pesos de cada clase
    """
    # Contar clases
    class_counts = np.bincount(y_train)
    
    # Calcular pesos (inverso de la frecuencia)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    return class_weights
