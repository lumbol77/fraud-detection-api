from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI(title="Fraud Detection ML API")

# 1. Load the Model and the Scaler
MODEL_PATH = "model/fraud_model.pkl"
SCALER_PATH = "model/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is Live and Scaled"}

@app.post("/predict")
def predict(data: dict):
    try:
        # 2. Extract the 5 features in the EXACT same order as train.py
        # Wallet sends 'transaction_hour', but our model was trained on 'Time'
        features = np.array([[
            data.get("v1", 0),
            data.get("v2", 0),
            data.get("v3", 0),
            data.get("amount", 0),
            data.get("transaction_hour", 0) 
        ]])

        # 3. Scale the input (Crucial!)
        scaled_features = scaler.transform(features)

        # 4. Make Prediction
        prediction = model.predict(scaled_features)[0]
        # Get probability for the "Confidence" score in your logs
        probability = model.predict_proba(scaled_features)[0][1]

        return {
            "is_fraud": bool(prediction),
            "confidence": round(float(probability), 4)
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}