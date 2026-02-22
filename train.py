import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("data/creditcard.csv")

# ========================= 
# 2. Split Features & Target
# =========================
X = data.drop("Class", axis=1)
y = data["Class"]

# =========================
# 3. Train-Test Split
# Stratify keeps fraud ratio balanced
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. Build Model
# class_weight handles imbalance
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

# =========================
# 5. Train Model
# =========================
model.fit(X_train, y_train)

# =========================
# 6. Evaluate Model
# =========================
y_pred = model.predict(X_test)
print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

# =========================
# 7. Save Model
# =========================
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fraud_model.pkl")

print("\nModel saved successfully in model/fraud_model.pkl")
