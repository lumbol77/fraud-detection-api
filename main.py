from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
#load train model
model=joblib.load('model/fraud_model.pkl')
app=FastAPI(title='Fraud Detection API')
#Define request body
class Transaction (BaseModel):
    features=list
    @app.get('/')
    def home():
        return{'message':'Fraud Detection API is runing'}
    @app.post('/predict')
    def predict(transaction:Transaction):
        data=np.array(transaction.features).reshape(1,-1)
        prediction=model.predict(data)
        return{
            'prediction':int(prediction[0]),
            'result':'fraud' if prediction[0]==1 else 'Not Fraud'

        }
        
