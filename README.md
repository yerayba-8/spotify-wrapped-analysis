# 🎵 Spotify Wrapped — Data Science & MLOps Pipeline

Análisis exploratorio y modelo predictivo de popularidad musical basado 
en 114.000 canciones reales de la API de Spotify.

## 🎯 Objetivo

Identificar qué características de audio predicen el éxito de una canción
y desplegar el modelo como API REST lista para producción.

## 📊 Resultados del modelo

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest ✅ | 10.0 | 13.72 | 0.15 |
| Gradient Boosting | 13.0 | 16.51 | 0.10 |
| Linear Regression | 15.0 | 18.70 | 0.00 |

**Insight clave:** Los audio features tienen baja correlación lineal con 
la popularidad (max. loudness: +0.05). La popularidad depende en gran medida 
de factores externos como marketing o playlists. El modelo Random Forest 
captura relaciones no lineales con un error medio de 0.08 puntos sobre 100.

## 🛠️ Stack técnico

| Capa | Tecnología |
|---|---|
| Análisis y EDA | Python, Pandas, Seaborn, Matplotlib |
| Modelado | Scikit-learn (Random Forest, Gradient Boosting) |
| Despliegue | FastAPI, Uvicorn |
| Persistencia | Joblib |
| Control de versiones | Git, GitHub |

## 📁 Estructura del proyecto

```text
spotify-wrapped-analysis/
├── data/
│   └── raw/                        # Dataset de Kaggle (no incluido por tamaño)
├── notebooks/
│   ├── 01_EDA.ipynb                # Análisis exploratorio y visualizaciones
│   └── 02_modelo.ipynb            # Entrenamiento, evaluación y comparativa
├── src/
│   ├── main.py                     # API FastAPI
│   ├── model.pkl                   # Modelo entrenado
│   └── label_encoder.pkl           # Encoder guardado
├── outputs/
│   └── figures/                    # Gráficas generadas
└── requirements.txt                # Dependencias del proyecto

## 🔍 Principales hallazgos del EDA

- **Géneros más populares:** pop, latin y dance-pop lideran en popularidad media
- **Correlación energy-acousticness:** -0.73 (canciones energéticas son poco acústicas)
- **Correlación energy-loudness:** +0.76 (más energía = más volumen)
- **Danceability-valence:** +0.48 (canciones bailables tienden a ser más alegres)
```

## ▶️ Cómo ejecutar la API localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/yerayba-8/spotify-wrapped-analysis.git
cd spotify-wrapped-analysis

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Arrancar la API
cd src
uvicorn main:app --reload

# 4. Documentación interactiva
# Abrir http://127.0.0.1:8000/docs
```

## 🧪 Ejemplo de predicción

```json
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "danceability": 0.8, "energy": 0.7, "loudness": -5.0,
       "speechiness": 0.05, "acousticness": 0.1,
       "instrumentalness": 0.0, "liveness": 0.1,
       "valence": 0.6, "tempo": 120.0,
       "explicit": 0, "genre": "pop"
     }'
```

**Respuesta:**
```json
{
  "genre": "pop",
  "popularity": 65.2,
  "label": "📈 Popular"
}
```

## 📈 Visualizaciones generadas

| Gráfica | Descripción |
|---|---|
| `top_genres_popularity.png` | Top 15 géneros por popularidad media |
| `top_artists_popularity.png` | Top 15 artistas globales |
| `popularity_distribution.png` | Distribución de popularidad |
| `correlation_heatmap.png` | Correlación entre audio features |
| `energy_vs_popularity.png` | Energía vs popularidad por género |
| `danceability_by_genre.png` | Danceability media por género |
| `feature_importance.png` | Importancia de features en Random Forest |
| `real_vs_predicted.png` | Valores reales vs predichos |
| `error_distribution.png` | Distribución del error |

## 👤 Autor

**Yeray Bueno De Arriba**  
Técnico de Sistemas en transición a Data & AI  
[LinkedIn](https://www.linkedin.com/in/yeray-bueno-de-arriba-807423202) · 
[GitHub](https://github.com/yerayba-8)

---
*Dataset: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) 
— maharshipandya (Kaggle)*

