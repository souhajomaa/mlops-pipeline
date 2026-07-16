import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.sklearn import log_model
import joblib
import os
import random
import json

# ========== CONFIGURATION ==========
DATASET_PATH = 'data/diabetes.csv'
MODEL_PATH = 'models/model.pkl'
METRICS_PATH = 'metrics.json'
SEUIL_ACCEPTATION = 0.75

# Charger le dataset
df = pd.read_csv(DATASET_PATH)
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# MLflow tracking
mlflow.set_experiment("mlops-pipeline-diabetes")

with mlflow.start_run():
    n_estimators = random.choice([50, 100, 150, 200])
    max_depth = random.choice([3, 5, 10, None])
    
    print(f"Run avec n_estimators={n_estimators}, max_depth={max_depth}")

    # Entrainement
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Logger dans MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", str(max_depth))
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("precision", float(precision))
    mlflow.log_metric("recall", float(recall))
    mlflow.log_metric("f1_score", float(f1))
    log_model(model, "model")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    run = mlflow.active_run()
    if run is not None:
        print(f"Run ID: {run.info.run_id}")

    # Sauvegarder le modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    
    # Sauvegarder les métriques pour DVC
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_estimators": n_estimators,
        "max_depth": str(max_depth),
        "dataset": "diabetes",
        "threshold": SEUIL_ACCEPTATION
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées dans {METRICS_PATH}")