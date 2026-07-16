from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="MLOps Pipeline API",
    description="API de prédiction du diabète",
    version="2.0.0"
)

MODEL_PATH = "models/model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 2,
                "glucose": 138,
                "blood_pressure": 62,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree": 0.127,
                "age": 47
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    result: str
    confidence: float

@app.get("/")
def root():
    return {"message": "MLOps Pipeline API - Diabète", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: DiabetesInput):
    try:
        features = np.array([[
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree,
            data.age
        ]])
        prediction = int(model.predict(features)[0])
        confidence = float(model.predict_proba(features)[0][prediction])
        
        return PredictionOutput(
            prediction=prediction,
            result="Diabétique" if prediction == 1 else "Non diabétique",
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))