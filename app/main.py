from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, Field # Added Field for validation
import logging
import os

# 1. Keep your existing Logging & Path logic
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Could not load model: {e}")
    model = None

app = FastAPI(title="Fraud Detection API")

# 2. IMPROVED: Named fields instead of just "list"
# This prevents the 500 errors you were seeing.
class Transaction(BaseModel):
    amount: float
    sender_balance: float
    receiver_balance: float
    hour_of_day: int
    is_international: int # 1 for True, 0 for False

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    logging.info("Prediction request received")

    # 3. Convert named fields into the array the model expects
    # This ensures the order is ALWAYS correct
    input_data = np.array([[
        transaction.amount,
        transaction.sender_balance,
        transaction.receiver_balance,
        transaction.hour_of_day,
        transaction.is_international
    ]])

    prediction = model.predict(input_data)
    logging.info(f"Prediction result: {prediction[0]}")

    return {
        "prediction": int(prediction[0]),
        "result": "Fraud" if prediction[0] == 1 else "Not Fraud"
    }