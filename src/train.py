import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Charger le dataset
df = pd.read_csv('data/dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow tracking
mlflow.set_experiment("mlops-pipeline")

with mlflow.start_run():
    # Parametres
    n_estimators = 100
    max_depth = 3

    # Entrainement
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Logger dans MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

    # Sauvegarder le modele
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    print("Modele sauvegarde dans models/model.pkl")
