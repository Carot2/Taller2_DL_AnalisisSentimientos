"""
Utilidades para el análisis de sentimiento en tweets
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def clean_text(text):
    """
    Función para limpiar el texto de los tweets
    
    Args:
        text (str): Texto del tweet a limpiar
        
    Returns:
        str: Texto limpio
    """
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Eliminar menciones (@usuario)
    text = re.sub(r'@\w+', '', text)
    
    # Eliminar hashtags (conservando la palabra sin el símbolo #)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Eliminar RT (retweet)
    text = re.sub(r'\brt\b', '', text)
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar todos los caracteres no ASCII (incluyendo emojis y caracteres especiales)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Eliminar puntuación
    text = re.sub(r'[^\w\s]', '', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def plot_history(history, title="Métricas de Entrenamiento"):
    """
    Graficar la historia de entrenamiento
    
    Args:
        history: Objeto history devuelto por model.fit()
        title (str): Título para la gráfica
    """
    plt.figure(figsize=(12, 4))
    
    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, show_confusion_matrix=True):
    """
    Evalúa el modelo y muestra métricas detalladas
    
    Args:
        model: Modelo entrenado de Keras
        X_test: Datos de prueba
        y_test: Etiquetas reales
        show_confusion_matrix (bool): Si se muestra la matriz de confusión
        
    Returns:
        dict: Diccionario con las métricas (accuracy, precision, recall, f1)
    """
    # Predicciones en los datos de prueba
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Imprimir métricas
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Mostrar matriz de confusión
    if show_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def apply_oversampling(df, random_state=42):
    """
    Aplica oversampling a la clase minoritaria
    
    Args:
        df (DataFrame): DataFrame con los datos
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        DataFrame: DataFrame balanceado
    """
    from sklearn.utils import resample
    
    # Separar clases
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]
    
    # Sobremuestreo de la clase minoritaria
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # muestreo con reemplazo
                                     n_samples=len(df_majority),
                                     random_state=random_state)
    
    # Combinar
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    # Barajar
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_balanced


def apply_smote(X_train, y_train, random_state=42):
    """
    Aplica SMOTE para equilibrar las clases
    
    Args:
        X_train: Características de entrenamiento
        y_train: Etiquetas de entrenamiento
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (X_train_resampled, y_train_resampled)
    """
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled


def plot_class_distribution(df, title="Distribución de Clases"):
    """
    Visualiza la distribución de clases
    
    Args:
        df (DataFrame): DataFrame con columna 'label'
        title (str): Título para la gráfica
    """
    # Distribución de las clases (sentimientos)
    sentiment_counts = df['label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimiento', 'Cantidad']
    sentiment_counts['Proporción (%)'] = (sentiment_counts['Cantidad'] / sentiment_counts['Cantidad'].sum() * 100).round(2)
    
    # Mostrar tabla
    print("Distribución de sentimientos:")
    print(sentiment_counts)
    
    # Visualización
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Sentimiento', y='Cantidad', data=sentiment_counts, palette='Dark2')
    
    # Añadir etiquetas
    for i, row in sentiment_counts.iterrows():
        ax.text(i, row['Cantidad'] / 2, f"{row['Cantidad']}\n({row['Proporción (%)']}%)",
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Sentimiento (0 = Negativo, 1 = Positivo)', fontsize=12)
    plt.ylabel('Cantidad de Tweets', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_text_length_by_sentiment(df, column='cleaned_tweet'):
    """
    Visualiza la distribución de longitud de texto por sentimiento
    
    Args:
        df (DataFrame): DataFrame con las columnas 'label' y la columna de texto
        column (str): Nombre de la columna que contiene el texto
    """
    # Calcular longitud de los tweets
    df['text_length'] = df[column].apply(len)
    
    # Crear gráfica de distribución
    plt.figure(figsize=(12, 5))
    
    # Histograma para tweets negativos (0)
    plt.subplot(1, 2, 1)
    neg_lengths = df[df['label'] == 0]['text_length']
    plt.hist(neg_lengths, bins=50, alpha=0.7, color='red')
    plt.axvline(np.percentile(neg_lengths, 95), color='black', linestyle='--', 
                label=f'Percentil 95: {np.percentile(neg_lengths, 95):.0f}')
    plt.title('Longitud de Tweets Negativos')
    plt.xlabel('Longitud del texto')
    plt.ylabel('Frecuencia')
    plt.legend()
    
    # Histograma para tweets positivos (1)
    plt.subplot(1, 2, 2)
    pos_lengths = df[df['label'] == 1]['text_length']
    plt.hist(pos_lengths, bins=50, alpha=0.7, color='green')
    plt.axvline(np.percentile(pos_lengths, 95), color='black', linestyle='--', 
                label=f'Percentil 95: {np.percentile(pos_lengths, 95):.0f}')
    plt.title('Longitud de Tweets Positivos')
    plt.xlabel('Longitud del texto')
    plt.ylabel('Frecuencia')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas descriptivas
    print("Estadísticas para tweets negativos:")
    print(neg_lengths.describe())
    print("\nEstadísticas para tweets positivos:")
    print(pos_lengths.describe())


