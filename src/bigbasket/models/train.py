# src/bigbasket/models/train.py
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

PROCESSED_DIR = os.getenv("OUT_DIR", "data/processed")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    X_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_train.parquet"))
    X_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "y_test.parquet"))
    return X_train, X_test, y_train.squeeze(), y_test.squeeze()

def train_and_save():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment("bigbasket_experiment")
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("Test acc:", acc)
        mlflow.log_metric("test_accuracy", acc)

        model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "rf_model")
        print("Saved model to", model_path)

if __name__ == "__main__":
    train_and_save()