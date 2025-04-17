"""
Script para evaluar modelos entrenados de análisis de sentimiento
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .model_bilstm_attention import AttentionLayer
from .utils import clean_text, evaluate_model
from .data_loader import load_data, prepare_data

def load_tokenizer(tokenizer_path):
    """
    Carga un tokenizer desde archivo JSON
    
    Args:
        tokenizer_path (str): Ruta al archivo JSON del tokenizer
        
    Returns:
        keras.preprocessing.text.Tokenizer: Tokenizer cargado
    """
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = f.read()
    
    return tokenizer_from_json(tokenizer_json)

def evaluate_saved_model(model_path, tokenizer_path=None, test_size=0.2, max_len=100):
    """
    Evalúa un modelo guardado en nuevos datos
    
    Args:
        model_path (str): Ruta al archivo .h5 del modelo
        tokenizer_path (str): Ruta al archivo JSON del tokenizer
        test_size (float): Proporción de datos para evaluación
        max_len (int): Longitud máxima de secuencia
        
    Returns:
        dict: Métricas de evaluación
    """
    # Cargar el modelo con objetos personalizados si es necesario
    custom_objects = {'AttentionLayer': AttentionLayer}
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Cargar datos
    df = load_data()
    
    # Si no se proporciona tokenizer_path, inferirlo del model_path
    if tokenizer_path is None:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('_model.h5', '')
        tokenizer_path = os.path.join(model_dir, f"{model_name}_tokenizer.json")
    
    # Verificar si existe el archivo del tokenizer
    if os.path.exists(tokenizer_path):
        # Cargar tokenizer existente
        tokenizer = load_tokenizer(tokenizer_path)
        
        # Preparar datos con tokenizer cargado
        _, X_test, _, y_test, _ = prepare_data(df, test_size=test_size)
        
        # Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test)
        
        return metrics
    else:
        # Si no hay tokenizer guardado, crear uno nuevo
        print(f"No se encontró tokenizer en {tokenizer_path}. Creando nuevo...")
        _, X_test, _, y_test, _ = prepare_data(df, test_size=test_size, max_len=max_len)
        
        # Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test)
        
        return metrics

def predict_sentiment(model_path, tokenizer_path, text, max_len=100):
    """
    Predice el sentimiento de un texto usando un modelo entrenado
    
    Args:
        model_path (str): Ruta al archivo .h5 del modelo
        tokenizer_path (str): Ruta al archivo JSON del tokenizer
        text (str): Texto para predecir
        max_len (int): Longitud máxima de secuencia
        
    Returns:
        float: Probabilidad de sentimiento positivo (0-1)
    """
    # Cargar el modelo con objetos personalizados si es necesario
    custom_objects = {'AttentionLayer': AttentionLayer}
    model = load_model(model_path, custom_objects=custom_objects)
    
    # Cargar tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Limpiar el texto
    cleaned_text = clean_text(text)
    
    # Convertir a secuencia
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # Padding
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    
    # Predecir
    prediction = model.predict(padded_sequence)[0][0]
    
    # Determinar sentimiento
    sentiment = "positivo" if prediction >= 0.5 else "negativo"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'confidence': float(confidence),
        'raw_score': float(prediction)
    }

def compare_models(model_paths, test_size=0.2):
    """
    Compara el rendimiento de varios modelos
    
    Args:
        model_paths (list): Lista de rutas a los modelos
        test_size (float): Proporción de datos para evaluación
        
    Returns:
        pandas.DataFrame: DataFrame con métricas comparativas
    """
    results = []
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('_model.h5', '')
        print(f"Evaluando modelo: {model_name}")
        
        # Evaluar modelo
        metrics = evaluate_saved_model(model_path, test_size=test_size)
        
        # Añadir nombre del modelo
        metrics['model'] = model_name
        results.append(metrics)
    
    # Crear DataFrame para comparación
    df_results = pd.DataFrame(results)
    
    # Reordenar columnas para mejor visualización
    df_results = df_results[['model', 'accuracy', 'precision', 'recall', 'f1']]
    
    return df_results

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Evaluación de modelos para análisis de sentimiento en tweets')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Ruta al archivo .h5 del modelo a evaluar')
    
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Ruta al archivo JSON del tokenizer (opcional)')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proporción de datos para prueba')
    
    parser.add_argument('--text', type=str, default=None,
                        help='Texto para predecir sentimiento (opcional)')
    
    parser.add_argument('--compare', action='store_true',
                        help='Comparar con otros modelos en el mismo directorio')
    
    args = parser.parse_args()
    
    # Si se proporciona texto, hacer predicción
    if args.text:
        result = predict_sentiment(args.model, args.tokenizer, args.text)
        print("\nPredicción de sentimiento:")
        print(f"Texto original: {result['text']}")
        print(f"Texto limpio: {result['cleaned_text']}")
        print(f"Sentimiento: {result['sentiment']}")
        print(f"Confianza: {result['confidence']:.4f}")
        print(f"Puntuación: {result['raw_score']:.4f}")
    
    # Si se solicita comparación
    elif args.compare:
        model_dir = os.path.dirname(args.model)
        model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) 
                      if f.endswith('_model.h5')]
        
        print(f"Comparando {len(model_files)} modelos...")
        df_comparison = compare_models(model_files, args.test_size)
        
        print("\nComparación de modelos:")
        print(df_comparison)
    
    # Evaluar un solo modelo
    else:
        metrics = evaluate_saved_model(args.model, args.tokenizer, args.test_size)
        
        print("\nResultados de evaluación:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
