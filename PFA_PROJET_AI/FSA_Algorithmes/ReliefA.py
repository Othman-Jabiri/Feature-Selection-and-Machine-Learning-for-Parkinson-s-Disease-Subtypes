
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# --- 0. Pré-requis : Définir les sous-types de référence (clusters_reference) ---
# These steps must be executed BEFORE this script or the results loaded from a file.
# For demonstration purposes, they are included here.

print("--- Préparation des données et définition des sous-types de référence ---")

# Load the original CSV dataset to get all data
data_path_features = "C:/Users/pc/Desktop/Features/PFA_PROJET_AI/data/pd_EEG_features.csv"
df_original_full = pd.read_csv(data_path_features)

# Remove non-numeric columns for normalization and FSA
drop_cols_for_features = []
if 'id' in df_original_full.columns: drop_cols_for_features.append('id')
if 'gender' in df_original_full.columns: drop_cols_for_features.append('gender')
# 'class' column is dropped because it's the original label, not the discovered subtype for unsupervised clustering reference
if 'class' in df_original_full.columns: drop_cols_for_features.append('class') 

df_features_for_fsa = df_original_full.drop(columns=drop_cols_for_features, errors='ignore')

# Min-Max Normalization
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_features_for_fsa), columns=df_features_for_fsa.columns)

# PCA with 4 principal components (for clustering reference)
pca_reference = PCA(n_components=4)
df_pca_reference = pd.DataFrame(pca_reference.fit_transform(df_normalized), columns=["PC1", "PC2", "PC3", "PC4"])

# KMeans Clustering with 4 clusters to define reference subtypes
kmeans_reference = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_reference = kmeans_reference.fit_predict(df_pca_reference)

print(f"Sous-types de référence générés. Nombre d'échantillons: {len(clusters_reference)}")
print(f"Nombre de caractéristiques originales: {df_normalized.shape[1]}")

# --- 1. ReliefA (Conceptual Simulation) - Supervised on DISCOVERED SUBTYPES ---
print("\n--- Application de la Simulation Conceptuelle de l'algorithme Relief ---")
print("Ceci est une approximation simplifiée et non une implémentation exacte de ReliefA/ReliefF.")

# The 'target' y for Relief concept will be the discovered subtypes (clusters_reference)
# This aligns with the goal of finding features that characterize the identified subtypes.
y_for_relief = clusters_reference

num_unique_classes = len(np.unique(y_for_relief))

if num_unique_classes == 2:
    print("  Détection de 2 classes : Calcul de la différence absolue des moyennes des caractéristiques entre les classes.")
    # Extract class labels (e.g., 0 and 1 if numeric)
    class_labels = np.unique(y_for_relief)
    class_0_mean = df_normalized[y_for_relief == class_labels[0]].mean()
    class_1_mean = df_normalized[y_for_relief == class_labels[1]].mean()
    
    # Importance is higher for larger absolute difference between class means
    relief_scores_concept = (class_0_mean - class_1_mean).abs()
else:
    print(f"  Détection de {num_unique_classes} classes : Utilisation de l'inverse de la variance intra-classe moyenne comme heuristique.")
    # For more than two classes, ReliefF logic is more complex (neighbors, etc.).
    # Here, we use a heuristic: features with low average intra-class variance
    # are generally good discriminators. So, the inverse of this variance gives a higher score.
    
    # Calculate variance of each feature within each class group, then average these variances.
    # We group df_normalized by y_for_relief (the discovered clusters)
    variance_intra_class = df_normalized.groupby(y_for_relief).var().mean()
    
    # Avoid division by zero by adding a small epsilon
    relief_scores_concept = 1 / (variance_intra_class + 1e-9)

# Sort features by their conceptual Relief score (most important first)
ranked_features_relief_concept = relief_scores_concept.sort_values(ascending=False)

print("\nTop 10 des caractéristiques selon Relief (Simulation Conceptuelle) :")
print(ranked_features_relief_concept.head(10).to_markdown(numalign="left", stralign="left"))


# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour Relief (Simulation Conceptuelle) ---")

# Define K values to test for the number of features
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Adjust this list if needed

best_ari_relief = -1.0
best_k_relief = -1
optimal_features_list_relief = [] # Stores the names of the features for the optimal K

for k in k_values_to_test:
    # Select the top K features based on Relief ranking
    current_selected_features_names = ranked_features_relief_concept.head(k).index.tolist()

    if not current_selected_features_names:
        continue # Skip if no features selected for this K

    # Filter normalized data with selected features
    X_selected_for_clustering = df_normalized[current_selected_features_names]

    # Apply KMeans on these selected features
    kmeans_fsa_test = KMeans(n_clusters=4, random_state=42, n_init=10) # Always 4 clusters
    clusters_fsa_test = kmeans_fsa_test.fit_predict(X_selected_for_clustering)

    # Calculate ARI
    ari = adjusted_rand_score(clusters_reference, clusters_fsa_test)

    # Update best ARI and optimal K
    if ari > best_ari_relief:
        best_ari_relief = ari
        best_k_relief = k
        optimal_features_list_relief = current_selected_features_names


# --- 3. Save Relief results ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

# Save ONLY the names of the optimal features AND their importance values into a CSV file.
if optimal_features_list_relief:
    # Select importance values for optimal features from the already sorted Series
    final_optimal_features_series = ranked_features_relief_concept.loc[optimal_features_list_relief]

    # Create a DataFrame with the list of optimal features and their importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'Relief_Importance_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "relieff_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False to not write the index
    print(f"\nLa liste des {len(optimal_features_list_relief)} caractéristiques optimales (avec leurs valeurs d'importance Relief) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour Relief.")


print(f"\n--- Résultats Optimaux pour Relief (Simulation Conceptuelle) ---")
print(f"Meilleur K pour Relief (selon ARI) : {best_k_relief}")
print(f"ARI le plus élevé : {best_ari_relief:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_relief)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_relief[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")

"""

import pandas as pd
import numpy as np

# --- 1. Chargement des données normalisées et de la cible ---

# Charger le dataset normalisé
# Assurez-vous que ce fichier est accessible (dans le même répertoire ou chemin complet correct)
X_normalized = pd.read_csv("C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_normalized.csv")

# Charger le dataset original juste pour récupérer la colonne 'class' (variable cible)
# Assurez-vous que ce fichier est accessible également
df_original = pd.read_csv("C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_features.csv")
y = df_original['class'] # La variable cible

print("Données normalisées et variable cible chargées.")
print(f"Forme des caractéristiques normalisées (X_normalized) : {X_normalized.shape}")
print(f"Forme de la variable cible (y) : {y.shape}")

# Dictionnaire pour stocker les classements de chaque FSA
fsa_rankings = {}

# --- 2. ReliefA (Simulation Conceptuelle) - Supervisé ---
print("\n--- Application de la Simulation Conceptuelle de l'algorithme Relief ---")
print("Ceci est une approximation simplifiée et non une implémentation exacte de ReliefA.")

# Vérifier le nombre de classes uniques dans la variable cible
num_unique_classes = y.nunique()

if num_unique_classes == 2:
    print("  Détection de 2 classes : Calcul de la différence absolue des moyennes des caractéristiques entre les classes.")
    # Extraire les noms des classes (par exemple, 0 et 1 si numériques)
    class_labels = y.unique()
    class_0_mean = X_normalized[y == class_labels[0]].mean()
    class_1_mean = X_normalized[y == class_labels[1]].mean()
    
    # L'importance est d'autant plus élevée que la différence absolue entre les moyennes des classes est grande
    relief_scores_concept = (class_0_mean - class_1_mean).abs()
else:
    print(f"  Détection de {num_unique_classes} classes : Utilisation de l'inverse de la variance intra-classe moyenne comme heuristique.")
    # Pour plus de deux classes, la logique de ReliefF est plus complexe (voisins, etc.).
    # Ici, nous utilisons une heuristique : les caractéristiques avec une faible variance moyenne AU SEIN des classes
    # sont généralement de bons discriminateurs. Donc, l'inverse de cette variance donne un score plus élevé.
    
    # Calculer la variance de chaque caractéristique au sein de chaque groupe de classe, puis la moyenne de ces variances.
    variance_intra_class = X_normalized.groupby(y).var().mean()
    
    # Éviter la division par zéro en ajoutant un petit epsilon
    relief_scores_concept = 1 / (variance_intra_class + 1e-9)

# Trier les caractéristiques par leur score Relief conceptuel (les plus importantes d'abord)
ranked_features_relief_concept = relief_scores_concept.sort_values(ascending=False)

fsa_rankings["Relief_Concept"] = ranked_features_relief_concept.index.tolist()

print("\nTop 10 des caractéristiques selon Relief (Simulation Conceptuelle) :")
print(ranked_features_relief_concept.head(10).to_markdown(numalign="left", stralign="left"))

# --- 3. Stockage des caractéristiques classées ---

# Pour Relief conceptuel, toutes les caractéristiques reçoivent un score.
# Nous allons sauvegarder toutes les caractéristiques classées et leurs scores.
ranked_features_relief_concept_df = ranked_features_relief_concept.to_frame(name='Relief_Concept_Score')

# Sauvegarder les caractéristiques classées dans un fichier CSV
output_filename = "ranked_features_by_Relief_Concept.csv"
ranked_features_relief_concept_df.to_csv(output_filename, index=True, header=True)

print(f"\n{len(ranked_features_relief_concept_df)} caractéristiques ont été classées par Relief (Simulation Conceptuelle).")
print(f"Les caractéristiques classées et leurs scores ont été sauvegardées dans '{output_filename}'.")

with open("ranked_features_relief_concept_df.txt", "w") as f:
     for feature_name in ranked_features_relief_concept_df.index:
         f.write(f"{feature_name}\n")
print(f"La liste des noms de caractéristiques sélectionnées a été sauvegardée dans 'ranked_features_relief_concept_df.txt'.")

"""