from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Iris Classifier API")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../artifacts/model.joblib")
model = joblib.load(model_path)

# Input schema
class Schema(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Mlops Assignment: Iris Classifier"}

@app.post("/predict")
def predict_species(data: Schema):
    input = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {
        "predicted_class": prediction
    }