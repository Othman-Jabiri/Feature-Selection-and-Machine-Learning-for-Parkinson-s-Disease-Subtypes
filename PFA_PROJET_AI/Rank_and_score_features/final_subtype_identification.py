import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# --- Configuration ---
results_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
final_optimal_features_file = os.path.join(results_dir, "final_optimal_features_names.txt")
data_path_features = "C:/Users/pc/Desktop/Features/PFA_PROJET_AI/data/pd_EEG_features.csv"

# Nombre de clusters attendu (basé sur le papier et vos choix précédents)
n_clusters_to_identify = 4

# --- 1. Préparation des données ---
print("--- Préparation des données pour l'identification finale des sous-types ---")

# Charger le dataset original complet pour obtenir toutes les caractéristiques
df_original_full = pd.read_csv(data_path_features)

# Supprimer les colonnes non pertinentes (ID, gender, class originale)
drop_cols_for_features = []
if 'id' in df_original_full.columns: drop_cols_for_features.append('id')
if 'gender' in df_original_full.columns: drop_cols_for_features.append('gender')
if 'class' in df_original_full.columns: drop_cols_for_features.append('class') # Laisser la 'class' originale pour la validation si nécessaire

df_features_for_clustering = df_original_full.drop(columns=drop_cols_for_features, errors='ignore')

# Normalisation standard des caractéristiques
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_features_for_clustering), columns=df_features_for_clustering.columns)

print(f"Données chargées et normalisées. Forme: {df_normalized.shape}")

# --- 2. Charger les caractéristiques optimales finales ---
print("\n--- Chargement des noms des caractéristiques optimales finales ---")

if not os.path.exists(final_optimal_features_file):
    print(f"ERREUR : Le fichier des caractéristiques optimales finales '{final_optimal_features_file}' est introuvable.")
    print("Veuillez exécuter 'validate_selected_features.py' pour générer ce fichier en premier.")
    exit()

# Lire le fichier texte contenant les noms des caractéristiques
with open(final_optimal_features_file, 'r') as f:
    optimal_feature_names = [line.strip() for line in f if line.strip()] # Lire chaque ligne et enlever les espaces

if not optimal_feature_names:
    print("ERREUR : Aucune caractéristique n'a été trouvée dans le fichier final_optimal_features_names.txt.")
    exit()

# Filtrer df_normalized pour ne garder que les caractéristiques optimales
# S'assurer que toutes les caractéristiques lues sont bien présentes dans df_normalized
# On retire les doublons au cas où, et on s'assure de l'ordre
final_features_for_model = [f for f in optimal_feature_names if f in df_normalized.columns]

if not final_features_for_model:
    print("ERREUR : Aucune des caractéristiques optimales lues n'est présente dans le dataset normalisé.")
    print("Veuillez vérifier les noms des colonnes dans le fichier pd_EEG_features.csv et les noms des caractéristiques sélectionnées.")
    exit()

X_final = df_normalized[final_features_for_model]

print(f"Nombre de caractéristiques optimales finales sélectionnées : {X_final.shape[1]}")
print(f"Jeu de données final pour le clustering : {X_final.shape}")


# --- 3. Entraînement du modèle KMeans final et attribution des sous-types ---
print("\n--- Entraînement du modèle KMeans final et attribution des sous-types ---")

kmeans_final = KMeans(n_clusters=n_clusters_to_identify, random_state=42, n_init=10) # n_init=10 est recommandé
final_subtypes = kmeans_final.fit_predict(X_final)

# Ajouter la colonne des sous-types au DataFrame original pour une analyse future facile
# Il est important que l'ordre des lignes corresponde !
df_original_full['Identified_Subtype'] = final_subtypes

print(f"Clustering final terminé. {n_clusters_to_identify} sous-types identifiés.")
print("Distribution des sous-types :")
print(pd.Series(final_subtypes).value_counts().sort_index())

# --- 4. Sauvegarde des résultats des sous-types ---
output_subtypes_csv = os.path.join(results_dir, "patients_with_identified_subtypes.csv")

# Sauvegarder le DataFrame original enrichi avec la colonne des sous-types
# Cela inclura toutes les colonnes originales + le sous-type identifié
df_original_full.to_csv(output_subtypes_csv, index=False)

print(f"\nLes patients avec leurs sous-types identifiés ont été sauvegardés dans '{output_subtypes_csv}'.")

print("\n--- Étape 1 (Identification Finale des Sous-types) terminée. ---")
print("Prochaine étape : Évaluation et Caractérisation des Sous-types Découverts.")