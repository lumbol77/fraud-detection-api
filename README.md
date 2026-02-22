# Fraud Detection API

A Machine Learning API for detecting fraudulent credit card transactions.

This project trains a classification model using historical transaction data and exposes it via a FastAPI REST API for real-time fraud prediction.

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Uvicorn

---

## Project Structure

```
fraud-detection-api/
│
├── app/
│   └── main.py            # FastAPI application
│
├── model/
│   └── model.pkl          # Trained ML model
│
├── train.py               # Model training script
├── requirements.txt       # Dependencies
├── README.md
├── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/lumbol77/fraud-detection-api.git
cd fraud-detection-api
```

Create virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

If you want to retrain:

```bash
python train.py
```

This will generate `model/model.pkl`.

---

## Run the API

```bash
uvicorn app.main:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

## Example Prediction Request

POST request to:

```
/predict
```

Example JSON:

```json
{
  "feature1": 0.12,
  "feature2": -1.45,
  ...
}
```

Response:

```json
{
  "prediction": "Fraud"
}
```

---

## Features

- Machine Learning classification model
- REST API with FastAPI
- Interactive Swagger UI
- Clean production-style structure
- Dataset excluded from repository

---

## Future Improvements

- Docker containerization
- Model performance monitoring
- Deployment to Render / Railway / AWS
- Authentication for API access

---

##  Author

Lumbo  
Aspiring Machine Learning Engineer
