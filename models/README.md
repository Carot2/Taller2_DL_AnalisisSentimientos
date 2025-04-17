# Modelos Entrenados

Este directorio contiene los modelos entrenados para el análisis de sentimiento de tweets.

## Archivos de Modelo

- `rnn_model.h5`: Modelo RNN básico entrenado.
- `lstm_model.h5`: Modelo LSTM estándar entrenado.
- `bilstm_attention_model.h5`: Modelo BiLSTM con mecanismo de atención entrenado.

## Detalles de Arquitectura

### RNN Básico
- Embedding (vocab_size, 64)
- SimpleRNN (64 unidades)
- Dense (1 unidad, activación sigmoid)

### LSTM Estándar
- Embedding (vocab_size, 64)
- LSTM (64 unidades)
- Dense (1 unidad, activación sigmoid)

### BiLSTM + Atención
- Embedding (vocab_size, 64)
- Bidirectional LSTM (64 unidades, return_sequences=True)
- Attention Layer (personalizada)
- Dense (1 unidad, activación sigmoid)

## Parámetros de Entrenamiento

Los modelos fueron entrenados con los siguientes parámetros:

- **Optimizer**: Adam (learning_rate=1e-3)
- **Loss**: Binary Crossentropy
- **Batch Size**: 128
- **Epochs**: 5
- **Validation Split**: 0.2

## Métricas de Rendimiento

Las métricas de rendimiento se actualizarán una vez completado el entrenamiento.

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| RNN    | -        | -         | -      | -        |
| LSTM   | -        | -         | -      | -        |
| BiLSTM+Atención | - | -       | -      | -        |

## Cómo Cargar un Modelo

```python
from tensorflow.keras.models import load_model
from src.model_bilstm_attention import AttentionLayer

# Para cargar el modelo BiLSTM+Atención
custom_objects = {'AttentionLayer': AttentionLayer}
model = load_model('models/bilstm_attention_model.h5', custom_objects=custom_objects)

# Para cargar RNN o LSTM
model_rnn = load_model('models/rnn_model.h5')
model_lstm = load_model('models/lstm_model.h5')
```
