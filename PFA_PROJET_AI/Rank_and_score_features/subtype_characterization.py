import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
results_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
patients_subtypes_file = os.path.join(
    results_dir, "patients_with_identified_subtypes.csv")
final_optimal_features_file = os.path.join(
    results_dir, "final_optimal_features_names.txt")
data_path_features = "C:/Users/pc/Desktop/Features/PFA_PROJET_AI/data/pd_EEG_features.csv"

# Nombre de clusters identifiés
n_identified_clusters = 4

# --- 1. Chargement des données et des sous-types identifiés ---
print("--- Chargement des données et des sous-types identifiés ---")

# Charger le DataFrame avec les sous-types identifiés
if not os.path.exists(patients_subtypes_file):
    print(
        f"ERREUR : Le fichier des patients avec sous-types '{patients_subtypes_file}' est introuvable.")
    print("Veuillez exécuter 'final_subtype_identification.py' pour le générer en premier.")
    exit()

df_patients = pd.read_csv(patients_subtypes_file)

# Charger les caractéristiques optimales finales pour les utiliser comme base du clustering
if not os.path.exists(final_optimal_features_file):
    print(
        f"ERREUR : Le fichier des caractéristiques optimales finales '{final_optimal_features_file}' est introuvable.")
    exit()

with open(final_optimal_features_file, 'r') as f:
    optimal_feature_names = [line.strip() for line in f if line.strip()]

if not optimal_feature_names:
    print("ERREUR : Aucune caractéristique optimale finale n'a été trouvée.")
    exit()

# Séparer les caractéristiques des autres colonnes pour le clustering
# IMPORTANT : Nous devons utiliser les caractéristiques qui ont réellement servi à l'entraînement du K-Means.
# Celles-ci doivent être normalisées et filtrées par optimal_feature_names.

# Recharger les caractéristiques et les normaliser de la même manière que précédemment
df_original_features = pd.read_csv(data_path_features)
drop_cols_initial = []
if 'id' in df_original_features.columns:
    drop_cols_initial.append('id')
if 'gender' in df_original_features.columns:
    drop_cols_initial.append('gender')
if 'class' in df_original_features.columns:
    drop_cols_initial.append('class')

df_features_only = df_original_features.drop(
    columns=drop_cols_initial, errors='ignore')
scaler = StandardScaler()
df_normalized_features_full = pd.DataFrame(scaler.fit_transform(
    df_features_only), columns=df_features_only.columns)

# S'assurer que les caractéristiques optimales sont bien présentes dans le DF normalisé
X_for_metrics = df_normalized_features_full[
    [f for f in optimal_feature_names if f in df_normalized_features_full.columns]
]

# Récupérer les sous-types identifiés
identified_subtypes = df_patients['Identified_Subtype']

print(
    f"Données chargées. {len(identified_subtypes)} échantillons et {X_for_metrics.shape[1]} caractéristiques utilisées pour le clustering.")

# --- 2. Évaluation Interne de la Qualité du Clustering ---
print("\n--- Évaluation Interne de la Qualité du Clustering ---")

if len(np.unique(identified_subtypes)) < 2:
    print("Moins de 2 clusters uniques. Impossible de calculer les métriques de qualité de clustering.")
else:
    try:
        silhouette_avg = silhouette_score(X_for_metrics, identified_subtypes)
        print(f"Coefficient de Silhouette : {silhouette_avg:.4f}")

        davies_bouldin_avg = davies_bouldin_score(
            X_for_metrics, identified_subtypes)
        print(
            f"Indice de Davies-Bouldin : {davies_bouldin_avg:.4f} (plus bas est mieux)")

        calinski_harabasz_avg = calinski_harabasz_score(
            X_for_metrics, identified_subtypes)
        print(
            f"Indice de Calinski-Harabasz : {calinski_harabasz_avg:.4f} (plus haut est mieux)")
    except Exception as e:
        print(f"Erreur lors du calcul des métriques de clustering : {e}")
        print(
            "Ceci peut arriver si le nombre d'échantillons dans un cluster est trop faible.")


# --- 3. Caractérisation des Sous-types ---
print("\n--- Caractérisation des Sous-types ---")

# Pour la caractérisation, il est souvent plus intuitif d'utiliser les données normalisées
# car elles ont une échelle comparable (0-1) et reflètent les "patterns" d'importance
# des caractéristiques.
df_characteristics = df_normalized_features_full.copy()
df_characteristics['Identified_Subtype'] = identified_subtypes

print("\nMoyennes des caractéristiques optimales par sous-type :")
# Calculer la moyenne de chaque caractéristique OPTIMALE par sous-type
# et afficher les 10 caractéristiques les plus distinctives pour chaque sous-type
# (celles avec les moyennes les plus élevées ou les plus basses par rapport à la moyenne globale ou aux autres clusters)

# Calculer la moyenne des caractéristiques optimales par sous-type
subtype_means = df_characteristics.groupby('Identified_Subtype')[
    optimal_feature_names].mean()
# Afficher toutes les moyennes si le nombre de features est gérable
print(subtype_means.to_markdown(numalign="left", stralign="left"))


# Analyser les caractéristiques les plus distinctives pour chaque sous-type
# Nous pouvons comparer la moyenne de chaque caractéristique dans un cluster à la moyenne globale.
# Une approche simple: pour chaque cluster, trouver les 5 caractéristiques avec les plus grandes/petites moyennes.
global_means = df_characteristics[optimal_feature_names].mean()

distinctive_features = {}
for subtype_id in sorted(df_characteristics['Identified_Subtype'].unique()):
    print(f"\n--- Sous-type {subtype_id} ---")

    # Caractéristiques avec les valeurs les plus élevées dans ce sous-type
    # Comparaison avec la moyenne globale pour voir ce qui est "élevé"
    relative_scores_high = subtype_means.loc[subtype_id] - global_means
    top_high_features = relative_scores_high.nlargest(10)
    print("Top 10 des caractéristiques avec les valeurs les plus ÉLEVÉES :")
    print(top_high_features.to_markdown(
        numalign="left", stralign="left", floatfmt=".4f"))

    # Caractéristiques avec les valeurs les plus basses dans ce sous-type
    # Positif si la moyenne du cluster est basse
    relative_scores_low = global_means - subtype_means.loc[subtype_id]
    top_low_features = relative_scores_low.nlargest(10)
    print("Top 10 des caractéristiques avec les valeurs les plus BASSES :")
    print(top_low_features.to_markdown(
        numalign="left", stralign="left", floatfmt=".4f"))

    distinctive_features[subtype_id] = {
        'top_high': top_high_features.index.tolist(),
        'top_low': top_low_features.index.tolist()
    }


# --- 4. Visualisation des Clusters ---
print("\n--- Visualisation des Clusters ---")

# Utilisation de PCA pour réduire les 300 caractéristiques à 2 composantes principales pour la visualisation
pca_viz = PCA(n_components=2, random_state=42)
# X_for_metrics contient les 300 caractéristiques optimales
components = pca_viz.fit_transform(X_for_metrics)

df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
df_pca['Identified_Subtype'] = identified_subtypes

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Identified_Subtype',
    palette=sns.color_palette("viridis", n_colors=n_identified_clusters),
    data=df_pca,
    legend='full',
    alpha=0.7
)
plt.title('Visualisation des Sous-types Identifiés (PCA 2D)')
plt.xlabel(
    f'Composante Principale 1 ({pca_viz.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(
    f'Composante Principale 2 ({pca_viz.explained_variance_ratio_[1]*100:.2f}%)')
plt.grid(True)
plt.savefig(os.path.join(results_dir, "subtype_pca_visualization.png"))
plt.show()

print("\nVisualisation des sous-types enregistrée.")

# --- Optionnel: Visualisation des moyennes des caractéristiques les plus importantes par sous-type ---
# Choisir quelques-unes des caractéristiques les plus importantes globalement pour les visualiser
# Ici, nous prenons les 5 premières caractéristiques de votre liste optimale
top_5_global_features = optimal_feature_names[:5]

if top_5_global_features:
    print(
        f"\nVisualisation des moyennes pour les 5 caractéristiques les plus importantes : {top_5_global_features}")

    df_plot_top_features = df_characteristics[[
        'Identified_Subtype'] + top_5_global_features]
    df_plot_melted = df_plot_top_features.melt(
        id_vars='Identified_Subtype', var_name='Feature', value_name='Normalized_Value')

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Feature', y='Normalized_Value',
                hue='Identified_Subtype', data=df_plot_melted, palette='viridis')
    plt.title('Moyenne des 5 Caractéristiques les plus Importantes par Sous-type')
    plt.xlabel('Caractéristique')
    plt.ylabel('Valeur Normalisée Moyenne')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "top_features_means_by_subtype.png"))
    plt.show()

print("\n--- Étape 2 (Évaluation et Caractérisation des Sous-types Découverts) terminée. ---")
print("Prochaine étape : Interprétation et Conclusion.")
