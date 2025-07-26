from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Iris Classifier API")

# Load the trained model from the correct path
model_path = os.path.join("artifacts", "model.joblib")
model = joblib.load(model_path)

# Define the input data schema using Pydantic
class Schema(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "MLOps Assignment: Iris Classifier"}

@app.post("/predict")
def predict_species(data: Schema):
    # Convert request data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return {
        "predicted_class": prediction
    }
