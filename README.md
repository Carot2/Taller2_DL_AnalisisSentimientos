# Análisis de Sentimiento en Tweets con RNN, LSTM y BiLSTM + Atención

Este proyecto implementa y compara diferentes arquitecturas de redes neuronales recurrentes para la clasificación de sentimientos en tweets.


## Índice
1. [Arquitecturas utilizadas](#arquitecturas-utilizadas)
   - [¿Qué es una RNN?](#qué-es-una-rnn)
   - [¿Qué es una LSTM?](#qué-es-una-lstm)
   - [¿Qué es una BiLSTM con atención?](#qué-es-una-bilstm-con-atención)
2. [Introducción a la problemática](#introducción-a-la-problemática)
3. [Exploración del dataset](#exploración-del-dataset)
   - [Descripción de las variables](#descripción-de-las-variables)
   - [Análisis de características y preprocesamiento](#análisis-de-las-características-y-preprocesamiento-de-datos)
4. [Implementación de la red neuronal](#implementación-de-la-red-neuronal)
5. [Estructuración del repositorio](#estructuración-del-repositorio)
6. [Instrucciones de uso](#instrucciones-de-uso-de-la-red-neuronal-y-rendimiento-obtenido)
   - [Requisitos previos](#requisitos-previos)
   - [Instalación](#instalación)
   - [Cómo entrenar el modelo](#cómo-entrenar-el-modelo-desde-trainpy)
   - [Cómo evaluar el modelo](#cómo-evaluar-el-modelo-usando-evaluatepy)
   - [Cómo hacer predicciones](#cómo-hacer-predicciones-con-predictpy)
7. [Rendimiento obtenido](#rendimiento-obtenido)
8. [Conclusiones](#conclusiones)


## ⚡️ Arquitecturas utilizadas

En este proyecto se implementaron y compararon tres tipos de arquitecturas de redes neuronales recurrentes para el análisis de sentimientos en Tweets:

### ¿Qué es una RNN?

![Red Neuronal Recurrente](https://github.com/Carot2/Taller2_DL_AnalisisSentimientos/blob/main/graphics/RNN.png)

Una Red Neuronal Recurrente (RNN) es un tipo de red diseñada para procesar secuencias de datos. A diferencia de una red neuronal tradicional, una RNN tiene una "memoria interna" que le permite recordar información de entradas anteriores, lo cual es útil en tareas como procesamiento de lenguaje natural. Sin embargo, las RNN simples suelen sufrir del problema de desvanecimiento del gradiente, lo que limita su capacidad para aprender dependencias a largo plazo.

### ¿Qué es una LSTM?

![Red Neuronal LSTM](https://mlarchive.com/wp-content/uploads/2024/04/New-Project-3-1-1024x607-1024x585.png)
La Long Short-Term Memory (LSTM) es una variante de la RNN diseñada para resolver los problemas de memoria de largo plazo. Utiliza compuertas (gate mechanisms) que controlan el flujo de información, permitiendo al modelo "recordar" o "olvidar" información según sea necesario. Esto la hace mucho más efectiva para tareas como clasificación de sentimientos en texto.


### ¿Qué es una BiLSTM con atención?

![BILSTM + ATTENTION](https://github.com/Carot2/Taller2_DL_AnalisisSentimientos/blob/main/graphics/BILSTM_ATTENTION.png)

Una BiLSTM (Bidirectional LSTM) procesa la secuencia tanto hacia adelante como hacia atrás, capturando contexto tanto previo como posterior en una oración. Esto mejora significativamente la comprensión semántica. Además, al añadir un mecanismo de **atención**, el modelo puede aprender a enfocarse en las palabras más relevantes del texto para tomar decisiones de clasificación, mejorando la interpretación y el rendimiento en tareas complejas como el análisis de sentimiento.

## ⚡️ Introducción a la problemática

### Aplicaciones del análisis de sentimiento en redes sociales

El **análisis de sentimiento** es una técnica clave del procesamiento de lenguaje natural (NLP) que busca identificar y extraer opiniones, emociones o actitudes expresadas en texto. En el contexto de redes sociales como Twitter, esta técnica tiene aplicaciones muy valiosas:

- **Monitoreo de la opinión pública**: detectar cómo reaccionan los usuarios ante eventos, marcas o personalidades.
- **Gestión de reputación**: las empresas pueden actuar rápidamente ante comentarios negativos o detectar embajadores de marca.
- **Marketing dirigido**: segmentación de usuarios según sus actitudes o emociones.
- **Sistemas de recomendación**: ofrecer contenido, productos o servicios personalizados en función del sentimiento expresado.
- **Análisis político o social**: comprender el sentimiento colectivo ante decisiones políticas o eventos sociales.

### Importancia de modelar secuencias y contexto lingüístico

A diferencia de tareas de clasificación más simples, en el análisis de sentimiento **el orden de las palabras y su contexto importan enormemente**. Por ejemplo:

> *"I don't like this"* no es lo mismo que *"I like this"*

Modelos como las RNN, LSTM y BiLSTM son esenciales porque:
- Permiten modelar **dependencias temporales** entre palabras.
- Capturan **el contexto lingüístico**, incluyendo negaciones, ironías o emociones implícitas.
- En el caso de BiLSTM, permiten analizar un texto **en ambas direcciones** (inicio → fin y fin → inicio).
- El mecanismo de **atención** mejora la capacidad del modelo para "enfocarse" en las palabras más relevantes dentro del tweet.

---

## ⚡️ Exploración del dataset

Para entrenar nuestros modelos, trabajamos con un conjunto de datos de tweets públicos extraído de [este repositorio de GitHub](https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv). El dataset contiene **31,962 tweets en inglés**, cada uno etiquetado con su **sentimiento binario**: positivo o negativo.

### **Descripción de las variables**

| Columna | Descripción |
|---------|-------------|
| `id`    | Identificador único del tweet. |
| `label` | Sentimiento asociado al tweet (`0`: negativo, `1`: positivo). |
| `tweet` | Texto completo del tweet, que puede contener menciones, hashtags, emojis, URLs y símbolos. |

se verificó que el dataset **no contiene valores nulos ni duplicados**, lo que facilita el proceso de preprocesamiento y entrenamiento de modelos sin requerir limpieza adicional por valores faltantes.

--

## 📊 Análisis de características y preprocesamiento de datos

A partir del análisis exploratorio del conjunto de datos, se pueden destacar las siguientes observaciones clave:

### 1. 🧮 Desbalanceo de clases

El dataset presenta un **fuerte desbalanceo de clases**, con:

- `29,720` tweets etiquetados como **negativos**
- `2,242` tweets etiquetados como **positivos**

Esto representa una proporción de aproximadamente **13:1 a favor de la clase negativa**, como puede observarse en la siguiente gráfica:
![Distribución de Tweets](https://github.com/Carot2/Taller2_DL_AnalisisSentimientos/blob/main/graphics/DistribucionTweets.png)


### 2. 🔠 Longitud de los tweets y tokens

- El análisis inicial muestra que:
  - El **95%** de los tweets negativos tiene una longitud menor o igual a **116 caracteres**
  - El **95%** de los tweets positivos no supera los **113 caracteres**

Esto justifica la elección de una **longitud máxima de 120 caracteres** como límite razonable para los modelos.

Posteriormente, al **tokenizar** los textos, se observó que el **95% de los tweets contiene como máximo 33 tokens**. Esta información se usó para definir el parámetro `max_len = 33` durante el padding, lo que permite una representación **uniforme, eficiente y sin pérdida de información**.
![Distribución de Tokens](https://github.com/Carot2/Taller2_DL_AnalisisSentimientos/blob/main/graphics/LongTokens.png)


### 3. 🧾 Palabras más frecuentes por clase

Se realizó un análisis de las palabras más frecuentes en tweets negativos y positivos. Los resultados muestran diferencias claras en el vocabulario utilizado, lo que indica que los modelos podrían aprender patrones relevantes para clasificar los tweets correctamente.

- En los tweets negativos predominan términos emocionales o expresivos como `love`, `happy`, `day`, `life`.

![PalabrasNegativas](https://github.com/Carot2/Taller2_DL_AnalisisSentimientos/blob/main/graphics/NubePalabrasNegativo.png)

- En los positivos (según etiqueta) se observan palabras relacionadas a política o ideologías como `trump`, `racist`, `libtard`.
![PalabrasPositivas](https://github.com/Carot2/Taller2_DL_AnalisisSentimientos/blob/main/graphics/NubePalabrasPositivo.png)

Esto puede deberse a sesgos en el dataset original, lo cual también debe tenerse en cuenta.

---

### 4. 🧪 Preprocesamiento aplicado

Para preparar los textos para el modelo, se aplicaron las siguientes transformaciones:

- Conversión a **minúsculas**
- Eliminación de **menciones** (`@user`)
- Eliminación de **hashtags**, manteniendo la palabra clave
- Remoción de **URLs**
- Eliminación de **emojis, caracteres especiales y puntuación**
- Reducción de **espacios múltiples**
- Tokenización y padding uniforme (`max_len = 33`)

---


Desde acá en construcción

### 5. ⚖️ Estrategias para abordar el desbalanceo ---

Debido al fuerte desbalance, se consideraron las siguientes estrategias:

- **Oversampling**: duplicar ejemplos de la clase minoritaria para equilibrar las proporciones.
- **SMOTE**: técnica de sobre-muestreo sintético que genera ejemplos nuevos de la clase minoritaria.
- **Class Weights**: ponderar la pérdida durante el entrenamiento para penalizar más los errores en la clase minoritaria.

En este proyecto, se experimentó principalmente con **oversampling** y `class_weight`, dependiendo del modelo implementado.







## Estructura del Repositorio

```
mi_proyecto_sentimiento/
├── notebooks/
│   ├── 01_exploracion.ipynb         # Exploración y preprocesamiento de datos
│   ├── 02_entrenamiento_RNN.ipynb   # Entrenamiento y evaluación del modelo RNN
│   ├── 03_entrenamiento_LSTM.ipynb  # Entrenamiento y evaluación del modelo LSTM
│   └── 04_BiLSTM_atencion.ipynb     # Entrenamiento y evaluación del modelo BiLSTM+Atención
├── src/
│   ├── data_loader.py               # Funciones para cargar y preprocesar los datos
│   ├── model_rnn.py                 # Implementación del modelo RNN
│   ├── model_lstm.py                # Implementación del modelo LSTM
│   ├── model_bilstm_attention.py    # Implementación del modelo BiLSTM+Atención
│   ├── train.py                     # Funciones de entrenamiento
│   ├── evaluate.py                  # Funciones de evaluación
│   └── utils.py                     # Funciones auxiliares
├── models/                          # Directorio para guardar modelos entrenados
│   ├── rnn_model.h5
│   ├── lstm_model.h5
│   ├── bilstm_attention_model.h5
│   └── README.md                    # Descripción de los modelos guardados
├── requirements.txt                 # Dependencias del proyecto
└── .gitignore                       # Archivos a ignorar en Git
```

## Requisitos y Dependencias

Para ejecutar este proyecto, necesitas tener instalado Python 3.8 o superior y las siguientes bibliotecas:

```
pip install -r requirements.txt
```

## Uso

### Carga y Exploración de Datos

El dataset se carga directamente desde una URL pública:

```python
import pandas as pd
import tensorflow as tf

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
csv_path = tf.keras.utils.get_file("twitter_sentiment.csv", url)
df = pd.read_csv(csv_path)
```

### Entrenamiento de Modelos

Cada modelo se puede entrenar ejecutando el notebook correspondiente:

```bash
jupyter notebook notebooks/02_entrenamiento_RNN.ipynb
jupyter notebook notebooks/03_entrenamiento_LSTM.ipynb
jupyter notebook notebooks/04_BiLSTM_atencion.ipynb
```

También se puede ejecutar desde la línea de comandos usando los scripts en `src/`:

```bash
python src/train.py --model rnn --epochs 5 --batch_size 128
python src/train.py --model lstm --epochs 5 --batch_size 128
python src/train.py --model bilstm_attention --epochs 5 --batch_size 128
```

### Evaluación de Modelos

Para evaluar un modelo entrenado:

```bash
python src/evaluate.py --model models/bilstm_attention_model.h5 --test_size 0.2
```



## Resultados

A continuación se presentan las métricas de rendimiento de cada modelo:

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| RNN    | X.XX     | X.XX      | X.XX   | X.XX     |
| LSTM   | X.XX     | X.XX      | X.XX   | X.XX     |
| BiLSTM+Atención | X.XX | X.XX  | X.XX   | X.XX     |

### Análisis Comparativo

(Aquí irá un análisis comparativo de los resultados una vez entrenados los modelos)

## Manejo del Desbalanceo de Clases

Se implementaron dos estrategias para abordar el desbalanceo de clases (29720 negativos vs 2242 positivos):

1. **Oversampling**: Duplicación aleatoria de ejemplos de la clase minoritaria.
2. **SMOTE**: Generación de ejemplos sintéticos para la clase minoritaria.

Los resultados muestran que... (Esto se completará con los resultados de los experimentos)



## ⚡️ Rendimiento obtenido



## 🧠Conclusiones




## Autores del Proyecto 🤓

Este proyecto fue desarrollado por:

* [Jeremías Pabón](https://github.com/jeremiaspabon) 
* [Gersón Julián Rincón](https://github.com/Julk-ui) 
* [Andrés Bravo](https://github.com/pipebravo10) 
* [Carolina Tobaria](https://github.com/Carot2) 