import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import json

# ========== CONFIGURATION ==========
DATASET_PATH = 'data/diabetes.csv'
MODEL_PATH = 'models/model.pkl'
METRICS_PATH = 'metrics.json'

def validate_model():
    print("=" * 50)
    print("SP-6 : Validation du modèle - Diabetes")
    print("=" * 50)
    
    # 1. Charger les données
    print("\n[1/4] Chargement des données...")
    df = pd.read_csv(DATASET_PATH)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # 2. Charger le modèle
    print("[2/4] Chargement du modèle...")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("❌ ERREUR : Modèle introuvable !")
        sys.exit(1)
    
    # 3. Prédire
    print("[3/4] Prédiction...")
    predictions = model.predict(X)
    
    # 4. Calculer les métriques
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    
    print(f"[4/4] Métriques calculées")
    
    # Lire le seuil depuis metrics.json
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
        seuil = metrics_data.get('threshold', 0.75)
    except:
        seuil = 0.75
    
    # 5. Résultat
    print("\n" + "=" * 50)
    print("RÉSULTAT DE VALIDATION")
    print("=" * 50)
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")
    print(f"Seuil F1  : {seuil*100:.0f}%")
    print("-" * 50)
    
    if f1 >= seuil:
        print("✅ MODÈLE VALIDÉ")
        print("=" * 50)
        return True
    else:
        print("❌ MODÈLE REJETÉ")
        print("=" * 50)
        return False

if __name__ == "__main__":
    resultat = validate_model()
    sys.exit(0 if resultat else 1)