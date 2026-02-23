from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import logging
import os

# -----------------------
# Logging Configuration
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import os
import joblib

# Get project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Build full model path
model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

print("Loading model from:", model_path)  # Debug line (remove later)

# Load model
model = joblib.load(model_path)
logging.info("Model loaded successfully")

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="Fraud Detection API")

# -----------------------
# Request Body Schema
# -----------------------
class Transaction(BaseModel):
    features: list

# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    logging.info("Prediction request received")

    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)

    logging.info(f"Prediction result: {prediction[0]}")

    return {
        "prediction": int(prediction[0]),
        "result": "Fraud" if prediction[0] == 1 else "Not Fraud"
    }