import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
results_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
combined_ranking_file = os.path.join(results_dir, "combined_optimal_features_ranking.csv")

# Chemin vers le dataset original (pour récupérer les caractéristiques)
data_path_features = "C:/Users/pc/Desktop/Features/PFA_PROJET_AI/data/pd_EEG_features.csv"

# Nombre de clusters attendu (4 sous-types de Parkinson)
n_clusters_expected = 4

# --- 0. Pré-requis : Définir les sous-types de référence (clusters_reference) ---
print("--- Préparation des données et définition des sous-types de référence ---")

# Charger le dataset original complet
df_original_full = pd.read_csv(data_path_features)

# Supprimer les colonnes non pertinentes (ID, gender, class originale)
drop_cols_for_features = []
if 'id' in df_original_full.columns: drop_cols_for_features.append('id')
if 'gender' in df_original_full.columns: drop_cols_for_features.append('gender')
if 'class' in df_original_full.columns: drop_cols_for_features.append('class') # Laisser la 'class' originale pour la validation si nécessaire

df_features_for_clustering = df_original_full.drop(columns=drop_cols_for_features, errors='ignore')

# Normalisation Min-Max (essentielle pour KMeans et PCA)
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_features_for_clustering), columns=df_features_for_clustering.columns)

# PCA pour la référence de clustering (avec n_clusters_expected composantes)
pca_reference = PCA(n_components=n_clusters_expected)
df_pca_reference = pd.DataFrame(pca_reference.fit_transform(df_normalized), columns=[f"PC{i+1}" for i in range(n_clusters_expected)])

# KMeans pour définir les sous-types de référence
kmeans_reference = KMeans(n_clusters=n_clusters_expected, random_state=42, n_init=10)
clusters_reference = kmeans_reference.fit_predict(df_pca_reference)

print(f"Sous-types de référence générés. Nombre d'échantillons: {len(clusters_reference)}")
print(f"Nombre de caractéristiques originales: {df_normalized.shape[1]}")

# --- 1. Charger le classement combiné des caractéristiques ---
print("\n--- Chargement du classement combiné des caractéristiques ---")

if not os.path.exists(combined_ranking_file):
    print(f"ERREUR : Le fichier de classement combiné '{combined_ranking_file}' est introuvable. Assurez-vous d'avoir exécuté 'combine_fsa_results.py' au préalable.")
    exit()

# MODIFICATION ICI: Lire le CSV en utilisant index_col=0 car le nom de la caractéristique est l'index
df_combined_ranking = pd.read_csv(combined_ranking_file, index_col=0)

# Extraire la liste des noms de caractéristiques classées depuis l'index
all_ranked_features = df_combined_ranking.index.tolist()

# Nettoyer la liste des noms de caractéristiques pour supprimer les valeurs NaN si elles existent
# Bien que index_col=0 devrait éviter cela, c'est une sécurité.
all_ranked_features = [f for f in all_ranked_features if pd.notna(f) and f in df_normalized.columns]

print(f"Classement combiné chargé. Nombre total de caractéristiques classées : {len(all_ranked_features)}")


# --- 2. Évaluation des performances de clustering avec différentes tailles de K ---
print("\n--- Évaluation des performances de clustering (KMeans) avec le classement combiné ---")

# Définir les K valeurs (nombre de caractéristiques) à tester
# Vous pouvez ajuster cette liste en fonction de vos besoins et de la granularité souhaitée
k_values_to_test = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100] # <--- MODIFIER ICI

ari_scores_combined = []
optimal_k_validation = -1
best_ari_validation = -1.0
best_selected_features = []

for k in k_values_to_test:
    if k > len(all_ranked_features):
        print(f"Avertissement : K={k} est supérieur au nombre total de caractéristiques ({len(all_ranked_features)}). Arrêt.")
        break
    
    # Sélectionner les top K caractéristiques du classement combiné
    current_selected_features = all_ranked_features[:k]
    
    # Filtrer le dataset normalisé avec les caractéristiques sélectionnées
    # Cette ligne est l'origine de l'erreur KeyError
    # S'assurer que toutes les features dans current_selected_features sont bien des colonnes de df_normalized
    valid_features = [f for f in current_selected_features if f in df_normalized.columns]
    
    if not valid_features:
        print(f"Avertissement : Aucune caractéristique valide trouvée pour K={k}. Saut de cette itération.")
        continue

    X_selected = df_normalized[valid_features]
    
    # Appliquer KMeans
    kmeans_test = KMeans(n_clusters=n_clusters_expected, random_state=42, n_init=10)
    clusters_test = kmeans_test.fit_predict(X_selected)
    
    # Calculer l'ARI
    ari = adjusted_rand_score(clusters_reference, clusters_test)
    ari_scores_combined.append({'K': k, 'ARI': ari})
    
    print(f"K = {k} caractéristiques -> ARI : {ari:.4f}")
    
    # Mettre à jour le meilleur ARI et le K optimal
    if ari > best_ari_validation:
        best_ari_validation = ari
        optimal_k_validation = k
        best_selected_features = valid_features # Sauvegarder les caractéristiques validées et utilisées

# Convertir les résultats en DataFrame pour une meilleure manipulation
df_ari_results = pd.DataFrame(ari_scores_combined)

print(f"\n--- Validation : Résultats Optimaux ---")
print(f"Meilleur K trouvé : {optimal_k_validation} caractéristiques")
print(f"ARI le plus élevé obtenu : {best_ari_validation:.4f}")
print(f"Top 10 des noms de caractéristiques pour le K optimal : {best_selected_features[:10]}...")

# --- 3. Sauvegarde des résultats de validation ---
output_validation_csv = os.path.join(results_dir, "validation_ari_results_combined_fsa.csv")
output_optimal_features_final_txt = os.path.join(results_dir, "final_optimal_features_names.txt")
output_optimal_features_final_csv = os.path.join(results_dir, "final_optimal_features_with_combined_score.csv")


df_ari_results.to_csv(output_validation_csv, index=False)
print(f"\nRésultats des scores ARI pour différentes valeurs de K sauvegardés dans '{output_validation_csv}'.")

# Sauvegarder la liste finale des caractéristiques optimales
if best_selected_features: # S'assurer qu'il y a des caractéristiques à sauvegarder
    with open(output_optimal_features_final_txt, "w") as f:
        for feature_name in best_selected_features:
            f.write(f"{feature_name}\n")
    print(f"Les noms des {optimal_k_validation} caractéristiques finales optimales ont été sauvegardés dans '{output_optimal_features_final_txt}'.")
else:
    print("\nAucune caractéristique optimale finale à sauvegarder (aucun K n'a produit de résultats valides).")


# Sauvegarder les caractéristiques finales optimales avec leurs scores combinés
if best_selected_features and not df_combined_ranking.empty: # S'assurer qu'il y a des caractéristiques et que le df n'est pas vide
    # Récupérer les scores combinés pour ces caractéristiques
    df_final_optimal_with_scores = df_combined_ranking.loc[best_selected_features].reset_index()
    df_final_optimal_with_scores.rename(columns={'index': 'Feature_Name'}, inplace=True) # Renommer la colonne d'index
    
    df_final_optimal_with_scores.to_csv(output_optimal_features_final_csv, index=False)
    print(f"Les {optimal_k_validation} caractéristiques finales optimales avec leurs scores combinés ont été sauvegardées dans '{output_optimal_features_final_csv}'.")
else:
    print("\nAucune caractéristique optimale finale à sauvegarder (aucun K n'a produit de résultats valides ou le classement combiné est vide).")


# --- 4. Visualisation des résultats ARI ---
print("\n--- Visualisation des résultats ARI ---")

if not df_ari_results.empty:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='K', y='ARI', data=df_ari_results, marker='o')
    plt.title('Performance ARI en fonction du nombre de caractéristiques sélectionnées (Classement Combiné)')
    plt.xlabel('Nombre de Caractéristiques (K)')
    plt.ylabel('Adjusted Rand Index (ARI)')
    plt.grid(True)
    plt.xticks(k_values_to_test, rotation=45)
    
    if optimal_k_validation != -1: # S'il y a eu un K optimal trouvé
        plt.axvline(x=optimal_k_validation, color='r', linestyle='--', label=f'Meilleur K = {optimal_k_validation} (ARI={best_ari_validation:.4f})')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ari_performance_combined_fsa.png"))
    plt.show()
    print("\nValidation terminée. Le graphique de performance ARI a été enregistré.")
else:
    print("\nAucun résultat ARI à visualiser (le DataFrame des résultats ARI est vide).")