"""
Implementación del modelo LSTM para análisis de sentimiento en tweets
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_lstm_model(vocab_size, embedding_dim=64, lstm_units=64, dropout_rate=0.3):
    """
    Crea un modelo LSTM para clasificación de sentimiento
    
    Args:
        vocab_size (int): Tamaño del vocabulario (número de palabras únicas)
        embedding_dim (int): Dimensionalidad de los embeddings
        lstm_units (int): Número de unidades en la capa LSTM
        dropout_rate (float): Tasa de dropout para regularización
        
    Returns:
        tensorflow.keras.Model: Modelo LSTM compilado
    """
    model = Sequential([
        # Capa de Embedding para convertir índices de palabras en vectores densos
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        
        # Capa LSTM - más efectiva que RNN simple para capturar dependencias a largo plazo
        LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        
        # Regularización con Dropout
        Dropout(dropout_rate),
        
        # Capa de salida para clasificación binaria
        Dense(1, activation='sigmoid')
    ])
    
    # Compilar el modelo
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    
    return model
