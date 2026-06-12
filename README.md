# 🎵 Spotify Wrapped — Data Science & MLOps Pipeline

Análisis exploratorio y modelo predictivo de popularidad musical 
basado en el dataset oficial de Spotify (114.000 canciones reales).

## 🎯 Objetivo

Identificar qué características de audio predicen el éxito de una canción
y desplegar el modelo como API REST para predicciones en tiempo real.

## 🛠️ Stack técnico

- **Análisis:** Python, Pandas, Seaborn, Matplotlib
- **Modelado:** Scikit-learn (Random Forest Regressor)
- **Despliegue:** FastAPI, Uvicorn, Joblib

## 📁 Estructura del proyecto

\`\`\`
data/        → dataset de Kaggle (no incluido por tamaño)
notebooks/   → EDA y entrenamiento
src/         → código de la API
outputs/     → gráficas y resultados
\`\`\`

## 📊 Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
— 114.000 canciones con audio features reales de la API de Spotify.

## 🚀 Resultados

*(se actualizará al completar el modelo)*

## ▶️ Cómo ejecutar la API

\`\`\`bash
pip install -r requirements.txt
uvicorn src.main:app --reload
\`\`\`

## 👤 Autor

Yeray Bueno De Arriba — [LinkedIn](https://www.linkedin.com/in/yeray-bueno-de-arriba-807423202)