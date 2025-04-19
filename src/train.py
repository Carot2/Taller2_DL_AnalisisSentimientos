"""
Script principal para entrenamiento de modelos de análisis de sentimiento
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .data_loader import load_data, prepare_data, get_class_weights
from .model_rnn import create_rnn_model
from .model_lstm import create_lstm_model
from .model_bilstm_attention import create_bilstm_attention_model, AttentionLayer
from .utils import plot_history, evaluate_model, plot_class_distribution

def train_model(model_type='bilstm_attention', 
                max_words=10000, 
                max_len=100, 
                embedding_dim=64, 
                units=64, 
                dropout_rate=0.3,
                batch_size=128, 
                epochs=5, 
                balance_method=None,
                use_class_weights=False,
                validation_split=0.2, 
                patience=3,
                save_dir='models'):
    """
    Entrena el modelo especificado en los datos de tweets
    
    Args:
        model_type (str): Tipo de modelo a entrenar ('rnn', 'lstm', o 'bilstm_attention')
        max_words (int): Tamaño máximo del vocabulario
        max_len (int): Longitud máxima de secuencia
        embedding_dim (int): Dimensionalidad de los embeddings
        units (int): Número de unidades en las capas recurrentes
        dropout_rate (float): Tasa de dropout para regularización
        batch_size (int): Tamaño del batch
        epochs (int): Número de épocas
        balance_method (str): Método de balanceo ('oversampling', 'smote' o None)
        use_class_weights (bool): Si se usan pesos de clase para balanceo
        validation_split (float): Proporción de datos para validación
        patience (int): Paciencia para early stopping
        save_dir (str): Directorio para guardar modelos
        
    Returns:
        tuple: (model, history)
    """
    # Crear directorio para guardar modelos si no existe
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Cargar datos
    print("Cargando y preprocesando datos...")
    df = load_data(balance_method=balance_method if balance_method == 'oversampling' else None)
    
    # Mostrar distribución de clases
    plot_class_distribution(df, title="Distribución de Clases Original")
    
    # Preparar datos
    test_size = 1 - validation_split
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(
        df, max_words=max_words, max_len=max_len, test_size=test_size,
        balance_method=balance_method if balance_method == 'smote' else None
    )
    
    # Mostrar info de los datos
    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"Tamaño del conjunto de prueba: {len(X_test)} muestras")
    print(f"Tamaño del vocabulario: {min(len(tokenizer.word_index) + 1, max_words)} palabras")
    
    # Calcular pesos de clase si se solicita
    class_weights = get_class_weights(y_train) if use_class_weights else None
    if use_class_weights:
        print("Pesos de clase:", class_weights)
    
    # Crear modelo según el tipo especificado
    print(f"Creando modelo {model_type}...")
    if model_type == 'rnn':
        model = create_rnn_model(max_words, embedding_dim, units, dropout_rate)
    elif model_type == 'lstm':
        model = create_lstm_model(max_words, embedding_dim, units, dropout_rate)
    elif model_type == 'bilstm_attention':
        model = create_bilstm_attention_model(
    vocab_size=max_words,
    max_len=max_len,
    embedding_dim=embedding_dim,
    lstm_units=units,
    dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    # Mostrar resumen del modelo
    model.summary()
    
    # Definir callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(save_dir, f"{model_type}_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entrenar modelo
    print(f"Entrenando modelo {model_type}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Graficar métricas de entrenamiento
    plot_history(history, title=f"Métricas de Entrenamiento - {model_type.upper()}")
    
    # Evaluar modelo
    print(f"\nEvaluando modelo {model_type}...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Guardar modelo y tokenizer
    model_path = os.path.join(save_dir, f"{model_type}_model.h5")
    tokenizer_path = os.path.join(save_dir, f"{model_type}_tokenizer.json")
    
    # Guardar tokenizer (para poder procesar nuevos textos)
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w') as f:
        f.write(tokenizer_json)
    
    print(f"Modelo guardado en: {model_path}")
    print(f"Tokenizer guardado en: {tokenizer_path}")
    
    return model, history, metrics

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos para análisis de sentimiento en tweets')
    
    parser.add_argument('--model', type=str, default='bilstm_attention',
                        choices=['rnn', 'lstm', 'bilstm_attention'],
                        help='Tipo de modelo a entrenar')
    
    parser.add_argument('--max_words', type=int, default=10000,
                        help='Tamaño máximo del vocabulario')
    
    parser.add_argument('--max_len', type=int, default=100,
                        help='Longitud máxima de secuencia')
    
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Dimensionalidad de los embeddings')
    
    parser.add_argument('--units', type=int, default=64,
                        help='Número de unidades en las capas recurrentes')
    
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Tasa de dropout para regularización')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Tamaño del batch')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Número de épocas')
    
    parser.add_argument('--balance', type=str, default=None,
                        choices=['oversampling', 'smote', "class_weights", None],
                        help='Método de balanceo de clases')
    
    parser.add_argument('--class_weights', action='store_true',
                        help='Usar pesos de clase para manejar desbalanceo')
    
    args = parser.parse_args()
    
    # Entrenar modelo con los argumentos proporcionados
    train_model(
        model_type=args.model,
        max_words=args.max_words,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        units=args.units,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        balance_method=args.balance,
        use_class_weights=args.class_weights
    )
