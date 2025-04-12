Claro! Aqui está a tradução para o inglês:

---

# 💻 Machine Learning with Stacking: Lead Classification and Medical Cost Prediction

This project showcases the application of **Machine Learning techniques using the Stacking model** in two distinct scenarios:

- **Marketing Lead Classification** to predict the conversion of potential customers.
- **Medical Cost Regression** to estimate expenses based on demographic and behavioral features.

Both solutions were developed with a focus on **data exploration**, **structured preprocessing**, and **robust model evaluation**.

---

## 📁 Data Structure

### Marketing Leads (`leads_cleaned.csv`)

- Problem: Binary classification (Converted or Not Converted).
- Features: Demographics, acquisition channels, interaction history, etc.
- Target: `Converted`

### Medical Costs (`healthcosts_cleaned.csv`)

- Problem: Continuous value regression (medical cost in dollars).
- Features: Age, sex, BMI, number of children, smoking status, region, etc.
- Target: `medical charges`

---

## ⚙️ Technologies Used

- **Language:** Python 3
- **Visualization Libraries:** `Seaborn`, `Plotly`, `Matplotlib`
- **Data Manipulation:** `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-learn`
- **Packaging and Environment:** `pipenv`, `joblib` (for saving/preprocessors)

---

## 🧪 Project Steps

### 1. Exploratory Data Analysis (EDA)

- Data visualization using `Seaborn` and `Plotly`
- Understanding the structure of the datasets

### 2. Data Preparation

- Splitting into numerical and categorical features
- Applying a saved `preprocessor` using `joblib`
- Splitting data into train and test sets (80/20)

### 3. Model Training

#### 🔵 Classification (Leads)

- Base Algorithms:
  - `SGDClassifier`
  - `SVC`
  - `DecisionTreeClassifier`
- Meta-model:
  - `LogisticRegression`
- Strategy:
  - `StackingClassifier` (Vanilla)

#### 🔴 Regression (Medical Costs)

- Base Algorithms:
  - `LinearRegression`
  - `ElasticNet`
  - `DecisionTreeRegressor`
- Meta-model:
  - `HuberRegressor`
- Strategy:
  - `StackingRegressor` (Vanilla)

---

## 📊 Model Evaluation

### Classification

- **Evaluated Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Visualization:**
  - Interactive confusion matrix (`Plotly`)
- **Additional Analysis:**
  - Average feature importance from base algorithms

### Regression

- **Evaluated Metrics:**
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
- **Visualization:**
  - Predictor variable importance

---

## 🔍 Model Explainability

- Individual predictions from base algorithms and final Stacking model (Classification)
- Direct comparison of results from each estimator

---

## 📦 How to Run

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Activate the virtual environment:**

```bash
pipenv shell
```

3. **Run the notebooks or scripts:**
   - `stacking_leads.ipynb` (or the respective `.py` script)
   - `stacking_costs.ipynb` (or the respective `.py` script)

---

## 🧠 Key Learnings

This project reinforces the importance of **feature engineering**, **curation of base models** in ensemble architectures, and **interpretation of results**, promoting robust and explainable models for real-world problems.

---

Claro! Aqui está a tradução completa para o espanhol:

---

# 💻 Machine Learning con Stacking: Clasificación de Leads y Predicción de Costos Médicos

Este proyecto presenta la aplicación de **técnicas de Aprendizaje Automático utilizando el modelo Stacking** en dos escenarios distintos:

- **Clasificación de Leads de Marketing** para predecir la conversión de clientes potenciales.
- **Regresión de Costos Médicos** para estimar gastos en función de características demográficas y de comportamiento.

Ambas soluciones fueron desarrolladas con enfoque en la **exploración de datos**, **preprocesamiento estructurado** y **evaluación robusta de los modelos**.

---

## 📁 Estructura de los Datos

### Leads de Marketing (`leads_cleaned.csv`)

- Problema: Clasificación binaria (Convertido o No Convertido).
- Características: Datos demográficos, canales de adquisición, historial de interacciones, etc.
- Variable objetivo: `Converted`

### Costos Médicos (`healthcosts_cleaned.csv`)

- Problema: Regresión de valor continuo (costo médico en dólares).
- Características: Edad, sexo, IMC, número de hijos, hábito de fumar, región, etc.
- Variable objetivo: `medical charges`

---

## ⚙️ Tecnologías Utilizadas

- **Lenguaje:** Python 3
- **Librerías de Visualización:** `Seaborn`, `Plotly`, `Matplotlib`
- **Manipulación de Datos:** `Pandas`, `NumPy`
- **Aprendizaje Automático:** `Scikit-learn`
- **Empaquetado y Entorno:** `pipenv`, `joblib` (para guardar/preprocesadores)

---

## 🧪 Etapas del Proyecto

### 1. Análisis Exploratorio de Datos (EDA)

- Visualización de datos usando `Seaborn` y `Plotly`
- Comprensión de la estructura de los conjuntos de datos

### 2. Preparación de los Datos

- Separación en características numéricas y categóricas
- Aplicación de un `preprocessor` guardado con `joblib`
- División de los datos en conjuntos de entrenamiento y prueba (80/20)

### 3. Entrenamiento de Modelos

#### 🔵 Clasificación (Leads)

- Algoritmos Base:
  - `SGDClassifier`
  - `SVC`
  - `DecisionTreeClassifier`
- Meta-modelo:
  - `LogisticRegression`
- Estrategia:
  - `StackingClassifier` (Vanilla)

#### 🔴 Regresión (Costos Médicos)

- Algoritmos Base:
  - `LinearRegression`
  - `ElasticNet`
  - `DecisionTreeRegressor`
- Meta-modelo:
  - `HuberRegressor`
- Estrategia:
  - `StackingRegressor` (Vanilla)

---

## 📊 Evaluación de Modelos

### Clasificación

- **Métricas Evaluadas:**
  - Precisión (Accuracy)
  - Precisión Positiva (Precision)
  - Sensibilidad (Recall)
  - F1-Score
- **Visualización:**
  - Matriz de confusión interactiva (`Plotly`)
- **Análisis Adicional:**
  - Importancia promedio de las características en los modelos base

### Regresión

- **Métricas Evaluadas:**
  - RMSE (Error Cuadrático Medio)
  - R² (Coeficiente de Determinación)
- **Visualización:**
  - Importancia de las variables predictoras

---

## 🔍 Explicabilidad del Modelo

- Predicciones individuales de los algoritmos base y del modelo final de Stacking (Clasificación)
- Comparación directa de los resultados de cada estimador

---

## 📦 Cómo Ejecutar

1. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

2. **Activar el entorno virtual:**

```bash
pipenv shell
```

3. **Ejecutar los notebooks o scripts:**
   - `stacking_leads.ipynb` (o el script `.py` correspondiente)
   - `stacking_costs.ipynb` (o el script `.py` correspondiente)

---

## 🧠 Aprendizajes Clave

Este proyecto refuerza la importancia de la **ingeniería de características**, la **selección cuidadosa de modelos base** en arquitecturas de ensamble, y la **interpretación de resultados**, promoviendo modelos robustos y explicables para problemas del mundo real.

---

# 💻 Machine Learning com Stacking: Classificação de Leads e Previsão de Custos Médicos

Este projeto apresenta a aplicação de técnicas de **Aprendizado de Máquina com o modelo Stacking** em dois contextos distintos:

- **Classificação de Leads de Marketing** para prever a conversão de clientes potenciais.
- **Regressão de Custos Médicos** para estimar despesas com base em características demográficas e comportamentais.

Ambas as soluções foram desenvolvidas com foco em **exploração de dados**, **preprocessamento estruturado** e **avaliação robusta dos modelos**.

---

## 📁 Estrutura dos Dados

### Leads de Marketing (`leads_cleaned.csv`)

- Problema: Classificação binária (Convertido ou Não Convertido).
- Atributos: Dados demográficos, canais de aquisição, histórico de interações, etc.
- Target: `Converted`

### Custos Médicos (`healthcosts_cleaned.csv`)

- Problema: Regressão de valor contínuo (custo médico em dólares).
- Atributos: Idade, sexo, IMC, número de filhos, tabagismo, região, etc.
- Target: `medical charges`

---

## ⚙️ Tecnologias Utilizadas

- **Linguagem:** Python 3
- **Bibliotecas de Visualização:** `Seaborn`, `Plotly`, `Matplotlib`
- **Manipulação de Dados:** `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-learn`
- **Empacotamento e Ambiente:** `pipenv`, `joblib` (para salvar/preprocessadores)

---

## 🧪 Etapas do Projeto

### 1. Análise Exploratória (EDA)

- Visualização de dados com `Seaborn` e `Plotly`
- Compreensão da estrutura dos datasets

### 2. Preparação dos Dados

- Separação em features numéricas e categóricas
- Aplicação de um `preprocessor` salvo com `joblib`
- Divisão dos dados em treino e teste (80/20)

### 3. Treinamento dos Modelos

#### 🔵 Classificação (Leads)

- Algoritmos Base:
  - `SGDClassifier`
  - `SVC`
  - `DecisionTreeClassifier`
- Meta-modelo:
  - `LogisticRegression`
- Estratégia:
  - `StackingClassifier` (Vanilla)

#### 🔴 Regressão (Custos Médicos)

- Algoritmos Base:
  - `LinearRegression`
  - `ElasticNet`
  - `DecisionTreeRegressor`
- Meta-modelo:
  - `HuberRegressor`
- Estratégia:
  - `StackingRegressor` (Vanilla)

---

## 📊 Avaliação dos Modelos

### Classificação

- **Métricas Avaliadas:**
  - Acurácia
  - Precisão
  - Recall
  - F1-Score
- **Visualização:**
  - Matriz de confusão interativa (`Plotly`)
- **Análise Adicional:**
  - Importância média das features dos algoritmos base

### Regressão

- **Métricas Avaliadas:**
  - RMSE (Root Mean Squared Error)
  - R² (Coeficiente de Determinação)
- **Visualização:**
  - Importância das variáveis preditoras

---

## 🔍 Explicabilidade dos Modelos

- Predições individuais dos algoritmos base e da predição final do modelo de Stacking (Classificação)
- Comparação direta dos resultados de cada estimador

---

## 📦 Como Executar

1. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

2. **Ative o ambiente virtual:**

```bash
pipenv shell
```

3. **Execute os notebooks ou scripts:**
   - `stacking_leads.ipynb` (ou o respectivo script `.py`)
   - `stacking_costs.ipynb` (ou o respectivo script `.py`)

---

## 🧠 Aprendizados

Este projeto reforça a importância da **engenharia de atributos**, da **curadoria dos modelos base** em arquiteturas de ensemble, e da **interpretação de resultados**, promovendo modelos robustos e interpretáveis para problemas do mundo real.
