import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# --- 0. Pré-requis : Définir les sous-types de référence (comme dans votre script principal) ---
# Ces étapes doivent être exécutées AVANT ce script LASSO ou les résultats stockés/chargés
# Pour des raisons de démonstration, je les inclus ici, mais idéalement,
# `clusters_reference` et `df_normalized` devraient être passés ou chargés d'un fichier.

print("--- Préparation des données et définition des sous-types de référence ---")

# Chargement du fichier CSV original pour obtenir toutes les données
data_path_features = "C:/Users/pc/Desktop/Features/PFA_PROJET_AI/data/pd_EEG_features.csv"
df_original_full = pd.read_csv(data_path_features)

# Suppression des colonnes non numériques pour la normalisation et la FSA
drop_cols_for_features = []
if 'id' in df_original_full.columns: drop_cols_for_features.append('id')
if 'gender' in df_original_full.columns: drop_cols_for_features.append('gender')
if 'class' in df_original_full.columns: drop_cols_for_features.append('class') # 'class' est supprimée pour la détection de sous-types

df_features_for_fsa = df_original_full.drop(columns=drop_cols_for_features, errors='ignore')

# Normalisation Min-Max
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_features_for_fsa), columns=df_features_for_fsa.columns)

# PCA avec 4 composantes principales
pca_reference = PCA(n_components=4)
df_pca_reference = pd.DataFrame(pca_reference.fit_transform(df_normalized), columns=["PC1", "PC2", "PC3", "PC4"])

# Clustering KMeans avec 4 clusters pour définir les sous-types de référence
# n_init est important pour la robustesse de KMeans
kmeans_reference = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_reference = kmeans_reference.fit_predict(df_pca_reference)

print(f"Sous-types de référence générés. Nombre d'échantillons: {len(clusters_reference)}")
print(f"Nombre de caractéristiques originales: {df_normalized.shape[1]}")

# --- 1. LASSO : Sélection de Caractéristiques & Évaluation ARI ---
print("\n--- Application de l'algorithme LASSO et évaluation ARI ---")

# LASSO est supervisé. La 'cible' y sera ici les sous-types de référence (clusters_reference).
# Cela signifie que LASSO sélectionnera les caractéristiques les plus discriminantes pour ces sous-types.
y_for_lasso = clusters_reference

# Utilisation de LassoCV pour trouver le meilleur alpha par validation croisée.
# max_iter augmenté pour la convergence
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(df_normalized, y_for_lasso)
optimal_alpha = lasso_cv.alpha_
print(f"Meilleur alpha trouvé par LASSO (LassoCV) : {optimal_alpha:.6f}")

# Entraîner un modèle Lasso avec le meilleur alpha
lasso_model = Lasso(alpha=optimal_alpha, random_state=42, max_iter=10000)
lasso_model.fit(df_normalized, y_for_lasso)

# Les coefficients des caractéristiques indiquent leur importance.
lasso_coefficients = pd.Series(lasso_model.coef_, index=df_normalized.columns)

# IMPORTANT : Trier les caractéristiques par la valeur ABSOLUE de leurs coefficients
# C'est cette série qui sera utilisée pour la sélection et l'affichage des valeurs.
ranked_features_lasso = lasso_coefficients.abs().sort_values(ascending=False)

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]

best_ari_lasso = -1.0
best_k_lasso = -1
optimal_features_list_lasso = [] # Cette liste stockera les noms des meilleures caractéristiques

# Itération pour trouver le K optimal via ARI
for k in k_values_to_test:
    # Sélectionner les top K caractéristiques (par ordre de valeur absolue de coefficient)
    current_selected_features = ranked_features_lasso.head(k).index.tolist()

    # S'assurer qu'il y a des caractéristiques sélectionnées
    if not current_selected_features:
        continue

    # Filtrer les données normalisées avec les caractéristiques sélectionnées
    X_selected = df_normalized[current_selected_features]

    # Appliquer KMeans sur ces caractéristiques sélectionnées
    kmeans_fsa_test = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters_fsa_test = kmeans_fsa_test.fit_predict(X_selected)

    # Calculer l'ARI
    ari = adjusted_rand_score(clusters_reference, clusters_fsa_test)

    # Mettre à jour le meilleur ARI et le K optimal
    if ari > best_ari_lasso:
        best_ari_lasso = ari
        best_k_lasso = k
        # Stocker la liste complète des noms des caractéristiques pour ce K optimal
        optimal_features_list_lasso = current_selected_features


# --- 2. Stockage des résultats du LASSO ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs de coefficient absolues dans un fichier CSV.
if optimal_features_list_lasso:
    # Sélectionner les caractéristiques optimales dans la série 'ranked_features_lasso'
    # qui contient déjà les valeurs absolues et est triée.
    # On utilise .loc[] pour s'assurer que l'ordre des features_list_lasso est respecté si nécessaire,
    # mais puisque ranked_features_lasso est déjà triée de la plus grande valeur absolue à la plus petite,
    # et que optimal_features_list_lasso est un sous-ensemble des premières, l'ordre sera conservé.
    final_optimal_features_series = ranked_features_lasso.loc[optimal_features_list_lasso]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs coefficients absolus
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'Absolute_Coefficient_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "lasso_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"La liste des {len(optimal_features_list_lasso)} caractéristiques optimales (avec leurs valeurs de coefficient absolues) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("Aucune caractéristique optimale à sauvegarder dans un fichier CSV.")


print(f"\n--- Résultats Optimaux pour LASSO ---")
print(f"Meilleur K pour LASSO (selon ARI) : {best_k_lasso}")
print(f"ARI le plus élevé : {best_ari_lasso:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_lasso)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_lasso[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")







































"""

import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
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

# Dictionnaire pour stocker les classements de chaque FSA (bien que nous n'utilisions que LASSO ici)
fsa_rankings = {}

# --- 2. LASSO (Supervisé) ---
print("\n--- Application de l'algorithme LASSO sur le dataset normalisé ---")

# Utilisation de LassoCV pour trouver le meilleur paramètre alpha par validation croisée.
# max_iter est augmenté pour assurer la convergence avec des datasets plus grands.
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_normalized, y)
print(f"Meilleur alpha trouvé par LASSO (LassoCV) : {lasso.alpha_:.4f}")

# Entraîner un modèle Lasso avec le meilleur alpha
lasso_model = Lasso(alpha=lasso.alpha_, random_state=42, max_iter=10000)
lasso_model.fit(X_normalized, y)

# Les coefficients des caractéristiques indiquent leur importance.
lasso_coefficients = pd.Series(lasso_model.coef_, index=X_normalized.columns)

# Trier les caractéristiques par la valeur absolue de leurs coefficients (les plus importants d'abord)
ranked_features_lasso = lasso_coefficients.abs().sort_values(ascending=False)

fsa_rankings["LASSO"] = ranked_features_lasso.index.tolist()

print("\nTop 10 des caractéristiques selon LASSO (par valeur absolue du coefficient) :")
print(ranked_features_lasso.head(10).to_markdown(numalign="left", stralign="left"))

# --- 3. Stockage des caractéristiques sélectionnées ---

# Définir un seuil pour considérer une caractéristique "sélectionnée" (coefficient non nul ou très proche de zéro)
# Vous pouvez ajuster ce seuil si vous voulez être plus ou moins strict.
selection_threshold = 1e-6
selected_features_df = ranked_features_lasso[ranked_features_lasso > selection_threshold].to_frame(name='LASSO_Coefficient_Absolute_Value')

# Sauvegarder les caractéristiques sélectionnées (celles avec un coefficient non nul) dans un fichier CSV
output_filename = "selected_features_by_LASSO.csv"
selected_features_df.to_csv(output_filename, index=True, header=True)

print(f"\nNombre de caractéristiques sélectionnées par LASSO (coefficient > {selection_threshold}) : {len(selected_features_df)}")
print(f"Les caractéristiques sélectionnées et leurs scores ont été sauvegardées dans '{output_filename}'.")





"""