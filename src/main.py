from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Carga de modelo y encoder ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent
model   = joblib.load(BASE_DIR / "modelo.pkl")
encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'explicit', 'genre_encoded'
]

# ── Schema de entrada ───────────────────────────────────────────────────
class SongFeatures(BaseModel):
    danceability:     float = Field(..., ge=0.0, le=1.0,   example=0.8)
    energy:           float = Field(..., ge=0.0, le=1.0,   example=0.7)
    loudness:         float = Field(..., ge=-60.0, le=5.0, example=-5.0)
    speechiness:      float = Field(..., ge=0.0, le=1.0,   example=0.05)
    acousticness:     float = Field(..., ge=0.0, le=1.0,   example=0.1)
    instrumentalness: float = Field(..., ge=0.0, le=1.0,   example=0.0)
    liveness:         float = Field(..., ge=0.0, le=1.0,   example=0.1)
    valence:          float = Field(..., ge=0.0, le=1.0,   example=0.6)
    tempo:            float = Field(..., ge=0.0, le=300.0, example=120.0)
    explicit:         int   = Field(..., ge=0,   le=1,     example=0)
    genre:            str   = Field(...,                   example="pop")

# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spotify Popularity Predictor",
    description="Predice la popularidad de una canción (0-100) a partir de sus audio features.",
    version="1.0.0"
)

# ── Endpoints ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "project": "Spotify Wrapped — ML Pipeline",
        "author":  "Yeray Bueno De Arriba",
        "docs":    "/docs"
    }

@app.get("/genres")
def get_genres():
    """Devuelve la lista de géneros disponibles para el modelo."""
    return {"genres": sorted(encoder.classes_.tolist())}

@app.post("/predict")
def predict(song: SongFeatures):
    """Predice la popularidad de una canción (0-100)."""

    # Validar género
    if song.genre not in encoder.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"Género '{song.genre}' no reconocido. Consulta /genres para ver los disponibles."
        )

    # Construir input
    genre_encoded = encoder.transform([song.genre])[0]

    input_df = pd.DataFrame([{
        'danceability':      song.danceability,
        'energy':            song.energy,
        'loudness':          song.loudness,
        'speechiness':       song.speechiness,
        'acousticness':      song.acousticness,
        'instrumentalness':  song.instrumentalness,
        'liveness':          song.liveness,
        'valence':           song.valence,
        'tempo':             song.tempo,
        'explicit':          song.explicit,
        'genre_encoded':     genre_encoded
    }])

    prediction = model.predict(input_df)[0]
    popularity  = round(float(np.clip(prediction, 0, 100)), 1)

    return {
        "genre":      song.genre,
        "popularity": popularity,
        "label":      _label(popularity)
    }

def _label(score: float) -> str:
    if score >= 75: return "🔥 Trending"
    if score >= 50: return "📈 Popular"
    if score >= 25: return "🎵 Nicho"
    return "💤 Underground"