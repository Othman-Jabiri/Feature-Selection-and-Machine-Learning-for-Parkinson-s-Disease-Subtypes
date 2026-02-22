import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# --- 1. Charger les données depuis une ligne brute (exemple simulé ici) ---
with open("donnees_brutes.txt", "r") as f:  # ton fichier contenant 1 ligne avec toutes les valeurs
    line = f.read().strip()

# Transformer la ligne en tableau de float
data = np.array([float(val.replace('E', 'e')) for val in line.split(',') if val != '']).reshape(1, -1)

# --- 2. Appliquer les deux normalisations ---
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

data_standard = standard_scaler.fit_transform(data)
data_robust = robust_scaler.fit_transform(data)

# --- 3. Comparaison : choisir la meilleure normalisation ---
# Critère simple : on préfère des valeurs centrées proches de 0, et sans trop de dispersion
def score_normalisation(normalized_data):
    mean = np.mean(normalized_data)
    std = np.std(normalized_data)
    return abs(mean) + abs(std - 1)  # score proche de 0 est meilleur

score_std = score_normalisation(data_standard)
score_robust = score_normalisation(data_robust)

# Choix du meilleur
if score_robust < score_std:
    best_data = data_robust
    best_method = "RobustScaler"
else:
    best_data = data_standard
    best_method = "StandardScaler"

print(f"✅ Meilleure méthode : {best_method} (score: {min(score_std, score_robust):.4f})")

# --- 4. Sauvegarder dans un fichier pour réutilisation ---
output_df = pd.DataFrame(best_data)
output_df.to_csv("donnees_normalisees.csv", index=False, header=False)

print("📁 Fichier 'donnees_normalisees.csv' sauvegardé avec succès.")
