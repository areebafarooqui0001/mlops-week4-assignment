from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Iris Classifier API")

# Load model
model = joblib.load("artifacts/model.joblib")

# Input schema
class Schema(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Mlops Assignment: Iris Classifier"}

@app.post("/predict/")
def predict_species(data: Schema):
    input = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {
        "predicted_class": prediction
    }