from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="MLOps Pipeline API",
    description="API de prediction du modele RandomForest - Iris Dataset",
    version="1.0.0"
)

# Charger le modele au demarrage
MODEL_PATH = "models/model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()

SPECIES = {0: "setosa", 1: "versicolor", 2: "virginica"}

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: int
    species: str
    confidence: float

@app.get("/")
def root():
    return {"message": "MLOps Pipeline API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])
        prediction = int(model.predict(features)[0])
        confidence = float(model.predict_proba(features)[0][prediction])
        return PredictionOutput(
            prediction=prediction,
            species=SPECIES[prediction],
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
