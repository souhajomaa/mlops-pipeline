import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import sys

# ========== CONFIGURATION ==========
SEUIL_ACCEPTATION = 0.80  # 80% minimum
# ===================================

def validate_model():
    """
    SP-6 : Valider le modèle avec un seuil d'acceptation
    Charge le modèle, teste sur les données, vérifie le seuil
    """
    
    print("=" * 50)
    print("SP-6 : Validation du modèle")
    print("=" * 50)
    
    # 1. Charger les données de test
    print("\n[1/4] Chargement des données...")
    df = pd.read_csv('data/dataset.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Charger le modèle entraîné
    print("[2/4] Chargement du modèle (models/model.pkl)...")
    try:
        model = joblib.load('models/model.pkl')
    except FileNotFoundError:
        print("❌ ERREUR : models/model.pkl introuvable !")
        print("   → Lance d'abord : python src/train.py")
        sys.exit(1)
    
    # 3. Faire des prédictions
    print("[3/4] Prédiction sur les données...")
    predictions = model.predict(X)
    
    # 4. Calculer l'accuracy
    accuracy = accuracy_score(y, predictions)
    print(f"[4/4] Accuracy calculée : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 5. Vérifier le seuil d'acceptation
    print("\n" + "=" * 50)
    print("RÉSULTAT DE VALIDATION")
    print("=" * 50)
    print(f"Seuil d'acceptation : {SEUIL_ACCEPTATION*100:.0f}%")
    print(f"Accuracy obtenue     : {accuracy*100:.2f}%")
    print("-" * 50)
    
    if accuracy >= SEUIL_ACCEPTATION:
        print("✅ MODÈLE VALIDÉ — Prêt pour le déploiement !")
        print("=" * 50)
        return True
    else:
        print("❌ MODÈLE REJETÉ — Accuracy insuffisante")
        print("   → Relancer l'entraînement avec d'autres paramètres")
        print("=" * 50)
        return False

if __name__ == "__main__":
    resultat = validate_model()
    
    # Code de retour pour le pipeline CI/CD (futur)
    if resultat:
        sys.exit(0)   # Succès
    else:
        sys.exit(1)   # Échec