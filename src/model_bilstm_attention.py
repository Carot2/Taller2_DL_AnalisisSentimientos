"""
Implementación del modelo BiLSTM con mecanismo de atención para análisis de sentimiento en tweets
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class AttentionLayer(Layer):
    """
    Capa personalizada de atención para resaltar las partes importantes de una secuencia
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Crear pesos entrenables para la capa de atención
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[1], 1),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Calcular el vector de atención
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        
        # Convertir a pesos mediante softmax (suman 1)
        alpha = K.softmax(e, axis=1)
        
        # Multiplicar cada vector oculto por su peso de atención
        context = inputs * alpha
        
        # Sumar los vectores ponderados por atención
        context = K.sum(context, axis=1)
        
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

def create_bilstm_attention_model(vocab_size, embedding_dim=64, lstm_units=64, dropout_rate=0.3):
    """
    Crea un modelo BiLSTM con mecanismo de atención para clasificación de sentimiento
    
    Args:
        vocab_size (int): Tamaño del vocabulario (número de palabras únicas)
        embedding_dim (int): Dimensionalidad de los embeddings
        lstm_units (int): Número de unidades en la capa LSTM
        dropout_rate (float): Tasa de dropout para regularización
        
    Returns:
        tensorflow.keras.Model: Modelo BiLSTM+Atención compilado
    """
    # Capa de entrada
    inputs = Input(shape=(None,))
    
    # Capa de Embedding para convertir índices de palabras en vectores densos
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    
    # Capa BiLSTM (procesa la secuencia en ambas direcciones)
    x = Bidirectional(LSTM(units=lstm_units, 
                          return_sequences=True,  # Retorna secuencia completa para la atención
                          dropout=dropout_rate, 
                          recurrent_dropout=dropout_rate))(x)
    
    # Mecanismo de atención
    x = AttentionLayer()(x)
    
    # Regularización con Dropout
    x = Dropout(dropout_rate)(x)
    
    # Capa de salida para clasificación binaria
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Crear y compilar el modelo
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    
    return model
