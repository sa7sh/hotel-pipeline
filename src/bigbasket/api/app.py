# src/bigbasket/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_DIR", "models/rf_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="BigBasket Model API")

class PredictRequest(BaseModel):
    """JSON with features as key:value"""
    data: dict

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.data])
    # apply the same feature engineering as training (not included here)
    preds = model.predict(df)
    probs = model.predict_proba(df).tolist() if hasattr(model, "predict_proba") else None
    return {"prediction": int(preds[0]), "probabilities": probs}

@app.get("/")
def root():
    return {"status": "ok"}