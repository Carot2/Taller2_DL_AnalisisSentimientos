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

### 🚀 Entrenamiento de Modelos

Para entrenar los modelos (RNN, LSTM o BiLSTM+Atención) tienes **dos opciones** según tu preferencia:

#### Opción 1: Usando Jupyter Notebooks (modo interactivo)

1. Abre una terminal (CMD, PowerShell o terminal de VSCode).
2. Ubícate en la carpeta raíz del proyecto:

   ```bash
   cd "C:\Users\USER\Documentos\Maestría\Deep Learning\Taller2_DL_AnalisisSentimientos"

Lanza el servidor de Jupyter Notebook:
```bash
   jupyter notebook
```
Se abrirá una ventana en tu navegador web. Dentro del navegador:
   Abre la carpeta notebooks/.

3. Ejecuta uno por uno los siguientes notebooks para entrenar cada modelo:

   02_entrenamiento_RNN.ipynb (para RNN)
   03_entrenamiento_LSTM.ipynb (para LSTM)
   04_BiLSTM_atencion.ipynb (para BiLSTM+Atención)

Corre las celdas dentro de cada notebook manualmente (Shift + Enter).

✅ Ideal si quieres ver gráficas de entrenamiento y outputs en tiempo real.

#### Opción 2: Usando scripts desde Terminal (modo directo)
Si prefieres entrenar los modelos directamente desde la terminal, sin abrir Jupyter, puedes ejecutar:

```bash
--RNN python -m src.train --model rnn --epochs 5 --batch_size 128
--LSTM python -m src.train --model lstm --epochs 5 --batch_size 128
--BiLSTM+Atención python -m src.train --model bilstm_attention --epochs 5 --batch_size 128
```

📌 Notas importantes:

* Los modelos entrenados se guardarán automáticamente en la carpeta /models/.
* El tokenizer utilizado también se guardará como archivo .json.
* Puedes personalizar el número de épocas (--epochs), tamaño de batch (--batch_size), balanceo (--balance) y si quieres usar pesos de clase (--class_weights).

## ⚙️ Estrategias de Balanceo de Clases

Por defecto, **los modelos se entrenan sin aplicar técnicas de balanceo de clases**.  
Esto significa que el dataset original, que presenta un fuerte desbalance (93% positivos vs 7% negativos), se utiliza tal cual para entrenar RNN, LSTM y BiLSTM+Atención.


## 🚀 Cómo entrenar modelos aplicando técnicas de balanceo

### 1. Entrenar con Oversampling

```bash
python -m src.train --model bilstm_attention --balance oversampling --epochs 5 --batch_size 128
python -m src.train --model lstm --balance oversampling --epochs 5 --batch_size 128
python -m src.train --model rnn --balance oversampling --epochs 5 --batch_size 128
```
### 2. Entrenar con SMOTE

```bash
python -m src.train --model bilstm_attention --balance smote --epochs 5 --batch_size 128
python -m src.train --model lstm --balance smote --epochs 5 --batch_size 128
python -m src.train --model rnn --balance smote --epochs 5 --batch_size 128
```
### 2. Entrenar con Class Weights

```bash
python -m src.train --model bilstm_attention --balance class_weights --epochs 5 --batch_size 128
python -m src.train --model lstm --balance class_weights --epochs 5 --batch_size 128
python -m src.train --model rnn --balance class_weights --epochs 5 --batch_size 128
```
## 📈 Evaluación y Predicción de Modelos

Una vez entrenados los modelos, puedes evaluarlos, comparar su rendimiento o hacer predicciones individuales de sentimientos usando el script `src/evaluate.py`.

Asegúrate de estar ubicado en la carpeta raíz del proyecto antes de ejecutar los siguientes comandos.

> **Nota:**  
> Los comandos de evaluación funcionan con cualquier modelo `.h5` entrenado, ya sea sobre datos balanceados (oversampling, SMOTE, class weights) o no balanceados.  
> Asegúrate de seleccionar el modelo correspondiente según la técnica que hayas utilizado en el entrenamiento.


---

### 🔎 Evaluar un solo modelo

Evalúa un modelo entrenado (`.h5`) sobre un conjunto de datos de prueba:

```bash
python -m src.evaluate --model models/bilstm_attention_model.h5 --test_size 0.2
python -m src.evaluate --model models/lstm_model.h5 --test_size 0.2
python -m src.evaluate --model models/rnn_model.h5 --test_size 0.2
```

### 🥽 Evaluar un modelo usando un tokenizer específico
Si deseas usar un tokenizer .json diferente al que se infiere por defecto:
```bash
python -m src.evaluate --model models/bilstm_attention_model.h5 --tokenizer models/bilstm_attention_tokenizer.json
```

### 🔬 Predecir el sentimiento de un texto personalizado
Puedes usar un modelo entrenado para predecir el sentimiento de un nuevo texto:
```bash
python -m src.evaluate --model models/bilstm_attention_model.h5 --tokenizer models/bilstm_attention_tokenizer.json --text "I love this product!"
```

### 📊 Comparar varios modelos
Compara automáticamente todos los modelos .h5 en la carpeta /models/, evaluando su precisión, recall, F1-score, etc.:
```bash
python -m src.evaluate --model models/bilstm_attention_model.h5 --compare
```
Nota: --model debe apuntar a un modelo ubicado dentro de la carpeta que contiene los demás modelos para comparar.

## 📈 Resultados

A continuación se presentan las métricas de rendimiento de cada modelo evaluado:
### Modelos con Oversampling
| Modelo          | Accuracy | Precision | Recall    | F1-Score |
|:----------------|:--------:|:---------:|:---------:|:--------:|
| BiLSTM+Atención | 0.497575 | 0.0799    | 0.5870    | 0.1407   |
| LSTM            | 0.600501 | 0.0877    | 0.5000    | 0.1492   |
| RNN             | 0.750821 | 0.0836    | 0.2566    | 0.1261   |

### Modelos con SMOTE

| Modelo          | Accuracy | Precision | Recall    | F1-Score |
|:----------------|:--------:|:---------:|:---------:|:--------:|
| BiLSTM+Atención | 0.663695 | 0.0872    | 0.4017    | 0.1434   |
| LSTM            | 0.689973 | 0.0785    | 0.3191    | 0.1261   |
| RNN             | 0.637885 | 0.0762    | 0.3750    | 0.1267   |

### Modelos con Class_weights

| Modelo          | Accuracy | Precision | Recall    | F1-Score |
|:----------------|:--------:|:---------:|:---------:|:--------:|
| BiLSTM+Atención | 0.926326 | 0.129032  | 0.00892   | 0.016701 |
| LSTM            | 0.929767 | 0.0000    | 0.00000   | 0.0000   |
| RNN             | 0.929923 | 0.0000    | 0.00000   | 0.0000   |

---

### 📊 Análisis Comparativo de Modelos

- El modelo **RNN** alcanzó el mayor **accuracy** (92.99%), sin embargo, **falló en detectar ejemplos positivos**, mostrando una precisión y recall de 0.0%.
- El modelo **LSTM** mejoró ligeramente la **precision** respecto a RNN, pero el **recall** sigue siendo muy bajo, indicando dificultades para identificar la clase minoritaria.
- El modelo **BiLSTM+Atención**, aunque presentó un leve descenso en **accuracy** (92.01%), logró un **mejor equilibrio** entre precisión y recall en comparación a los anteriores, gracias al mecanismo de atención.

> **Conclusión preliminar:** El modelo BiLSTM+Atención ofrece un mejor compromiso entre sensibilidad (recall) y precisión para un dataset desbalanceado.

---

## ⚖️ Manejo del Desbalanceo de Clases

Para abordar el severo desbalance en el dataset original (29,720 negativos vs 2,242 positivos), se aplicaron las siguientes estrategias:

1. **Oversampling**: Duplicación aleatoria de ejemplos de la clase minoritaria para igualar el número de ejemplos de la clase mayoritaria.
2. **SMOTE (Synthetic Minority Over-sampling Technique)**: Generación sintética de nuevos ejemplos de la clase minoritaria en el espacio de características.

Ambas estrategias fueron evaluadas en diferentes etapas del proyecto.  
Los resultados sugieren que **combinar técnicas de balanceo con arquitecturas avanzadas** (como BiLSTM con atención) mejora la capacidad del modelo para detectar correctamente la clase minoritaria.

---

## ⚡️ Rendimiento obtenido

Tras entrenar y evaluar las tres arquitecturas propuestas (RNN, LSTM y BiLSTM+Atención) en el análisis de sentimiento de tweets, se observaron los siguientes resultados:

- **RNN**: Alcanzó el mayor accuracy general (~92.99%), pero con incapacidad para identificar correctamente la clase minoritaria (recall = 0.00%).
- **LSTM**: Mejoró marginalmente la capacidad de predicción de ejemplos positivos en comparación con la RNN, pero el recall aún resultó muy bajo (~0.0045).
- **BiLSTM+Atención**: Logró un mejor balance entre precisión y sensibilidad, incrementando el recall (~2.90%) y el F1-Score (~4.84%), sacrificando ligeramente el accuracy.

> **Conclusión de rendimiento**: Aunque la RNN obtuvo un mayor accuracy global, el modelo BiLSTM+Atención demostró ser superior en la detección de la clase minoritaria, lo cual es crucial en datasets altamente desbalanceados como el evaluado.

## 🧠 Conclusiones

- El análisis de sentimiento en datasets desbalanceados requiere algo más que optimizar el accuracy general; es fundamental mejorar métricas como el **recall** y el **F1-score**.
- Modelos básicos como la **RNN** tienden a sesgarse hacia la clase mayoritaria, ignorando ejemplos de clases minoritarias.
- **LSTM** mejora levemente la capacidad de generalización, pero no es suficiente por sí sola en presencia de desbalance severo.
- **BiLSTM combinada con mecanismos de Atención** demostró ser la arquitectura más efectiva, logrando capturar contexto bidireccional y priorizar palabras clave relevantes en los tweets.
- La implementación de técnicas de balanceo como **oversampling** y **class_weight** son imprescindibles para mejorar la detección de ejemplos de la clase minoritaria.
- Para proyectos futuros, se recomienda explorar técnicas adicionales como **focal loss** y **estrategias de data augmentation textual** para seguir mejorando la sensibilidad del modelo ante ejemplos minoritarios.

> **En conclusión:** BiLSTM+Atención, junto con estrategias de balanceo, proporciona una solución más robusta y adecuada para el análisis de sentimientos en entornos de datos desbalanceados.


## Autores del Proyecto 🤓

Este proyecto fue desarrollado por:

* [Jeremías Pabón](https://github.com/jeremiaspabon) 
* [Gersón Julián Rincón](https://github.com/Julk-ui) 
* [Andrés Bravo](https://github.com/pipebravo10) 
* [Carolina Tobaria](https://github.com/Carot2) 