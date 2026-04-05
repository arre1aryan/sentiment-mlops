import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Sentiment Analysis API")
Instrumentator().instrument(app).expose(app)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    proba = model.predict_proba([request.text])[0]
    confidence = round(float(proba.max()), 4)
    label = "positive" if proba[1] > 0.5 else "negative"
    return PredictionResponse(label=label, confidence=confidence)