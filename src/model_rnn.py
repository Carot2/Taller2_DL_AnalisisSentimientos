"""
Implementación del modelo RNN básico para análisis de sentimiento en tweets
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_rnn_model(vocab_size, embedding_dim=64, rnn_units=64, dropout_rate=0.3):
    """
    Crea un modelo RNN básico para clasificación de sentimiento
    
    Args:
        vocab_size (int): Tamaño del vocabulario (número de palabras únicas)
        embedding_dim (int): Dimensionalidad de los embeddings
        rnn_units (int): Número de unidades en la capa RNN
        dropout_rate (float): Tasa de dropout para regularización
        
    Returns:
        tensorflow.keras.Model: Modelo RNN compilado
    """
    model = Sequential([
        # Capa de Embedding para convertir índices de palabras en vectores densos
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        
        # Capa RNN básica
        SimpleRNN(units=rnn_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        
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
