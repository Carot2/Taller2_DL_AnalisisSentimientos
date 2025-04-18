""" Módulo para cargar y preprocesar los datos de tweets """
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

def prepare_data(df, max_words=10000, max_len=100, test_size=0.2, val_size=0.1, balance_method=None, random_state=42):
    """
    Prepara los datos para entrenamiento, incluyendo tokenización y padding
    
    Args:
        df (pandas.DataFrame): DataFrame con tweets limpios
        max_words (int): Tamaño máximo del vocabulario
        max_len (int): Longitud máxima de cada secuencia
        test_size (float): Proporción de datos para testing
        val_size (float): Proporción de datos para validación (del conjunto de entrenamiento)
        balance_method (str): Método de balanceo ('smote' o None)
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Diccionario con los conjuntos de datos y el tokenizer
    """
    # Primero dividir en train y test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # Mantener proporción de clases
    )
    
    # Luego dividir train en train y validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size/(1-test_size),  # Ajustar proporción
        random_state=random_state,
        stratify=train_df['label']  # Mantener proporción de clases
    )
    
    # Tokenización
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_df['cleaned_tweet'])
    
    # Convertir textos a secuencias
    X_train_seq = tokenizer.texts_to_sequences(train_df['cleaned_tweet'])
    X_val_seq = tokenizer.texts_to_sequences(val_df['cleaned_tweet'])
    X_test_seq = tokenizer.texts_to_sequences(test_df['cleaned_tweet'])
    
    # Padding
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Obtener etiquetas
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    # Aplicar SMOTE si se solicita (solo a los datos de entrenamiento)
    if balance_method == 'smote':
        X_train_pad, y_train = apply_smote(X_train_pad, y_train, random_state=random_state)
    
    # Empaquetar todo en un diccionario para facilitar su uso
    data = {
        'X_train': X_train_pad,
        'y_train': y_train,
        'X_val': X_val_pad,
        'y_val': y_val,
        'X_test': X_test_pad,
        'y_test': y_test,
        'tokenizer': tokenizer,
        'vocab_size': min(max_words, len(tokenizer.word_index) + 1)
    }
    
    return data

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar y balancear datos
    df = load_data(balance_method='oversampling')
    
    # Preparar datos para entrenamiento
    data = prepare_data(df, balance_method=None)  # No usar SMOTE aquí si ya aplicamos oversampling
    
    # Verificar distribución de clases
    print(f"Conjunto de entrenamiento: {np.bincount(data['y_train'])}")
    print(f"Conjunto de validación: {np.bincount(data['y_val'])}")
    print(f"Conjunto de prueba: {np.bincount(data['y_test'])}")
    
    # Los datos están listos para ser utilizados en el entrenamiento de modelos
    # X_train, y_train = data['X_train'], data['y_train']
    # X_val, y_val = data['X_val'], data['y_val']
    # X_test, y_test = data['X_test'], data['y_test']
    # vocab_size = data['vocab_size']