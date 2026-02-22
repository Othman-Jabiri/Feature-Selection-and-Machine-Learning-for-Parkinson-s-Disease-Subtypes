import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
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


# --- 1. CSFA (Correlation-based Feature Sorting) - Supervisé par les sous-types découverts ---
print("\n--- Application de l'algorithme CSFA (Corrélation avec les sous-types découverts) ---")

# Concaténer les caractéristiques normalisées avec les sous-types découverts (clusters_reference)
# pour calculer les corrélations.
# Il est crucial que les index de df_normalized et clusters_reference correspondent.
df_for_corr = df_normalized.copy()
df_for_corr['discovered_subtype'] = clusters_reference # Ajouter la colonne des sous-types comme cible

# Calculer la corrélation de Pearson de chaque caractéristique avec la colonne 'discovered_subtype' (cible)
# Nous prenons la valeur absolue pour classer par la force de la corrélation.
# Assurez-vous que la colonne cible est numérique (ce qui est le cas pour les clusters de KMeans).
correlations_with_target = df_for_corr.corr(numeric_only=True)['discovered_subtype'].abs().drop('discovered_subtype')

# Trier les caractéristiques par leur score de corrélation (les plus corrélées d'abord)
ranked_features_csfa = correlations_with_target.sort_values(ascending=False)

print("\nTop 10 des caractéristiques selon CSFA (par corrélation absolue avec les sous-types découverts) :")
print(ranked_features_csfa.head(10).to_markdown(numalign="left", stralign="left"))


# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour CSFA ---")

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Ajustez cette liste si nécessaire

best_ari_csfa = -1.0
best_k_csfa = -1
optimal_features_list_csfa = [] # Stockera les noms des caractéristiques pour le K optimal

for k in k_values_to_test:
    # Sélectionner les top K caractéristiques basées sur le classement CSFA
    current_selected_features_names = ranked_features_csfa.head(k).index.tolist()

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
    if ari > best_ari_csfa:
        best_ari_csfa = ari
        best_k_csfa = k
        optimal_features_list_csfa = current_selected_features_names


# --- 3. Sauvegarde des résultats du CSFA ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs d'importance dans un fichier CSV.
if optimal_features_list_csfa:
    # Sélectionner les valeurs d'importance pour les caractéristiques optimales
    final_optimal_features_series = ranked_features_csfa.loc[optimal_features_list_csfa]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'CSFA_Importance_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "csfa_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"\nLa liste des {len(optimal_features_list_csfa)} caractéristiques optimales (avec leurs valeurs d'importance CSFA) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour CSFA.")


print(f"\n--- Résultats Optimaux pour CSFA ---")
print(f"Meilleur K pour CSFA (selon ARI) : {best_k_csfa}")
print(f"ARI le plus élevé : {best_ari_csfa:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_csfa)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_csfa[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")



"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

# --- 2. CSFA (Correlation-based Feature Sorting) - Supervisé ---
print("\n--- Application de l'algorithme CSFA (Corrélation avec la cible) sur le dataset normalisé ---")

# Concaténer les caractéristiques normalisées et la cible pour calculer les corrélations
# Il est important de s'assurer que les index de X_normalized et y correspondent
df_for_corr = pd.concat([X_normalized, y.reset_index(drop=True)], axis=1)

# Calculer la corrélation de Pearson de chaque caractéristique avec la colonne 'class' (cible)
# Nous prenons la valeur absolue pour classer par la force de la corrélation, qu'elle soit positive ou négative.
correlations_with_target = df_for_corr.corr()['class'].abs().drop('class') # Exclure la corrélation de 'class' avec elle-même

# Trier les caractéristiques par leur score de corrélation (les plus corrélées d'abord)
ranked_features_csfa = correlations_with_target.sort_values(ascending=False)

fsa_rankings["CSFA"] = ranked_features_csfa.index.tolist()

print("\nTop 10 des caractéristiques selon CSFA (par corrélation absolue avec la cible) :")
print(ranked_features_csfa.head(10).to_markdown(numalign="left", stralign="left"))

# --- 3. Stockage des caractéristiques classées ---

# CSFA classe toutes les caractéristiques par leur corrélation.
# Nous allons sauvegarder toutes les caractéristiques classées et leurs scores.
ranked_features_csfa_df = ranked_features_csfa.to_frame(name='CSFA_Correlation_Absolute_Value')

# Sauvegarder les caractéristiques classées dans un fichier CSV
output_filename = "ranked_features_by_CSFA.csv"
ranked_features_csfa_df.to_csv(output_filename, index=True, header=True)

print(f"\n{len(ranked_features_csfa_df)} caractéristiques ont été classées par CSFA.")
print(f"Les caractéristiques classées et leurs scores ont été sauvegardées dans '{output_filename}'.")
# Si vous voulez juste la liste des noms de caractéristiques sélectionnées dans un fichier texte:
with open("ranked_features_by_CSFA.txt", "w") as f:
     for feature_name in ranked_features_csfa_df.index:
         f.write(f"{feature_name}\n")
print(f"La liste des noms de caractéristiques sélectionnées a été sauvegardée dans 'ranked_features_by_CSFA.txt'.")
"""