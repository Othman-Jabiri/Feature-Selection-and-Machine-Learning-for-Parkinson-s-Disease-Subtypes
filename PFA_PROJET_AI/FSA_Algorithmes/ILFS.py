
import pandas as pd
import numpy as np
from PyIFS import InfFS # Importation depuis le package PyIFS
import math # Not strictly used, but kept if the original script implies its use
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# --- 0. Pré-requis : Définir les sous-types de référence (clusters_reference) ---
print("--- Préparation des données et définition des sous-types de référence ---")

# Charger le dataset original pour toutes les caractéristiques
data_path_features = "C:/Users/pc/Desktop/Features/PFA_PROJET_AI/data/pd_EEG_features.csv"
df_original_full = pd.read_csv(data_path_features)

# Supprimer les colonnes non pertinentes pour la sélection de caractéristiques
drop_cols_for_features = []
if 'id' in df_original_full.columns: drop_cols_for_features.append('id')
if 'gender' in df_original_full.columns: drop_cols_for_features.append('gender')
# 'class' est la cible originale; nous la supprimons ici car nous construisons des sous-types non supervisés
if 'class' in df_original_full.columns: drop_cols_for_features.append('class')

df_features_for_fsa = df_original_full.drop(columns=drop_cols_for_features, errors='ignore')

# Normalisation Min-Max des caractéristiques
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_features_for_fsa), columns=df_features_for_fsa.columns)

# PCA avec 4 composantes principales (pour la référence de clustering)
pca_reference = PCA(n_components=4)
df_pca_reference = pd.DataFrame(pca_reference.fit_transform(df_normalized), columns=["PC1", "PC2", "PC3", "PC4"])

# Clustering KMeans avec 4 clusters pour définir les sous-types de référence (notre "cible" synthétique)
kmeans_reference = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_reference = kmeans_reference.fit_predict(df_pca_reference)

print(f"Sous-types de référence générés. Nombre d'échantillons: {len(clusters_reference)}")
print(f"Nombre de caractéristiques originales: {df_normalized.shape[1]}")


# --- 1. Application de l'algorithme ILFS (Infinite Latent Feature Selection) ---
print("\n--- Application de ILFS (Infinite Latent Feature Selection) ---")

X_np = df_normalized.values # Caractéristiques normalisées
# La cible pour ILFS est les sous-types découverts par KMeans
y_for_ilfs = clusters_reference

# Initialisation de l'algorithme ILFS
ilfs_model = InfFS()

# Appliquer ILFS. `supervision=True` car nous fournissons `y_for_ilfs`.
# `alpha` est un paramètre de régularisation dans ILFS. 0.5 est une valeur par défaut courante.
ranking_ilfs_indices, weights_all_features_ilfs = ilfs_model.infFS(X_np, y_for_ilfs, alpha=0.5, supervision=True, verbose=False)

# Récupérer les noms des caractéristiques dans l'ordre de classement
feature_names = df_normalized.columns
ranked_features_names_ilfs = feature_names[ranking_ilfs_indices].tolist()

# Créer une Series pour les scores triés, essentielle pour l'évaluation ARI
# Les scores de ILFS sont les `weights_all_features_ilfs`
# On s'assure que les scores sont positifs car ce sont des importances.
# S'ils peuvent être négatifs (ce qui n'est pas typique pour des poids d'importance), on peut prendre abs().
ranked_features_ilfs_scores = pd.Series(weights_all_features_ilfs, index=feature_names)
# Trier par ordre décroissant des scores
ranked_features_ilfs_scores = ranked_features_ilfs_scores.sort_values(ascending=False)


print("\nTop 10 des caractéristiques selon ILFS :")
print(ranked_features_ilfs_scores.head(10).to_markdown(numalign="left", stralign="left"))


# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour ILFS ---")

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Ajustez cette liste si nécessaire

best_ari_ilfs = -1.0
best_k_ilfs = -1
optimal_features_list_ilfs = [] # Stockera les noms des caractéristiques pour le K optimal

for k in k_values_to_test:
    # Sélectionner les top K caractéristiques basées sur le classement ILFS
    current_selected_features_names = ranked_features_ilfs_scores.head(k).index.tolist()

    if not current_selected_features_names:
        continue # Passer si aucune caractéristique sélectionnée pour ce K

    # Filtrer les données normalisées (qui sont samples x features) avec les caractéristiques sélectionnées
    X_selected_for_clustering = df_normalized[current_selected_features_names]

    # Appliquer KMeans sur ces caractéristiques sélectionnées
    kmeans_fsa_test = KMeans(n_clusters=4, random_state=42, n_init=10) # Toujours 4 clusters
    clusters_fsa_test = kmeans_fsa_test.fit_predict(X_selected_for_clustering)

    # Calculer l'ARI
    ari = adjusted_rand_score(clusters_reference, clusters_fsa_test)

    # Mettre à jour le meilleur ARI et le K optimal
    if ari > best_ari_ilfs:
        best_ari_ilfs = ari
        best_k_ilfs = k
        optimal_features_list_ilfs = current_selected_features_names


# --- 3. Sauvegarde des résultats du ILFS ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs d'importance dans un fichier CSV.
if optimal_features_list_ilfs:
    # Sélectionner les valeurs d'importance pour les caractéristiques optimales
    final_optimal_features_series = ranked_features_ilfs_scores.loc[optimal_features_list_ilfs]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'ILFS_Importance_Value': final_optimal_features_series.values
    })

    optimal_features_csv_path = os.path.join(output_dir, "ilfs_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"\nLa liste des {len(optimal_features_list_ilfs)} caractéristiques optimales (avec leurs valeurs d'importance ILFS) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour ILFS.")


print(f"\n--- Résultats Optimaux pour ILFS ---")
print(f"Meilleur K pour ILFS (selon ARI) : {best_k_ilfs}")
print(f"ARI le plus élevé : {best_ari_ilfs:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_ilfs)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_ilfs[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")


"""
import pandas as pd
import numpy as np
from PyIFS import InfFS # Importation depuis le package PyIFS
import math
# --- 1. Chargement des données normalisées et de la cible ---

X_normalized = pd.read_csv("C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_normalized.csv")
df_original = pd.read_csv("C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_features.csv")
y = df_original['class']

print("Données normalisées et variable cible chargées.")
print(f"Forme des caractéristiques normalisées (X_normalized) : {X_normalized.shape}")
print(f"Forme de la variable cible (y) : {y.shape}")

# Dictionnaire pour stocker les classements
fsa_rankings = {}

# --- 2. Application de l'algorithme ILFS (Infinite Latent Feature Selection) ---

print("\n--- Application de ILFS (Infinite Latent Feature Selection) ---")

# Conversion en numpy
X_np = X_normalized.values
y_np = y.values if len(set(y)) > 1 else None  # Si classification binaire ou multiclasses

# Initialisation de l'algorithme ILFS
ilfs_model = InfFS()

# Appliquer ILFS (supervisé si y est fourni)
ranking_ilfs, weights_ilfs = ilfs_model.infFS(X_np, y_np, alpha=0.5, supervision=True, verbose=False)


# Récupérer les noms des caractéristiques dans l'ordre
feature_names = X_normalized.columns
ranked_features_ilfs = feature_names[ranking_ilfs]

# Stocker dans le dictionnaire
fsa_rankings["ILFS"] = ranked_features_ilfs.tolist()

# Afficher les top 10
print("\nTop 10 des caractéristiques selon ILFS :")
for i in range(10):
    print(f"{i+1}. {ranked_features_ilfs[i]} (poids: {weights_ilfs[ranking_ilfs[i]]:.4f})")

# --- 3. Sauvegarde des résultats ---

# Créer DataFrame
ranked_features_ilfs_df = pd.DataFrame({
    'Feature': ranked_features_ilfs,
    'ILFS_Score': weights_ilfs[ranking_ilfs]
})

# Sauvegarde CSV
output_filename_ilfs = "ranked_features_by_ILFS.csv"
ranked_features_ilfs_df.to_csv(output_filename_ilfs, index=False)
print(f"\nLes caractéristiques classées et leurs scores ILFS ont été sauvegardées dans '{output_filename_ilfs}'.")

# Sauvegarde des noms dans fichier texte
with open("ranked_features_ilfs.txt", "w") as f:
    for feature in ranked_features_ilfs:
        f.write(f"{feature}\n")
print("La liste des noms de caractéristiques sélectionnées a été sauvegardée dans 'ranked_features_ilfs.txt'.")
"""