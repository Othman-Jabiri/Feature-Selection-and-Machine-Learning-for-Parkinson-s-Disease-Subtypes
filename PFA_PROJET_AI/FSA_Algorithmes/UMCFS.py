import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Lars
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# --- Fonctions internes de UMCFS (inchangées sauf la correction) ---
def construct_affinity_matrix(X, k=5, t=None):
    A = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    A = 0.5 * (A + A.T)
    D = cdist(X, X, 'euclidean')
    if t is None:
        # Calculer t comme la moyenne des distances pour les paires connectées dans A
        t = np.mean(D[A.toarray().astype(bool)]) if np.any(A.toarray()) else 1.0 # Éviter division par zéro
    W = np.exp(-D ** 2 / (2 * t ** 2))
    W *= A.toarray()
    return W

def laplacian_eigenmap(W, n_components=5):
    L, D_diag = laplacian(W, normed=False, return_diag=True)
    # Correction ici : L est déjà un numpy.ndarray dense, donc pas besoin de .toarray()
    eigvals, eigvecs = eigh(L, np.diag(D_diag)) 
    
    # Retourne les n_components premiers vecteurs propres non triviaux
    # On trie par valeurs propres croissantes et prend les n_eigen suivantes.
    sorted_indices = np.argsort(eigvals)
    # Skip the first trivial eigenvector (index 0 after sorting)
    return eigvecs[:, sorted_indices[1:n_components+1]]

def umcfs(X, feature_names, k=5, n_eigen=5, ratio=1.0):
    W = construct_affinity_matrix(X, k)
    Y = laplacian_eigenmap(W, n_eigen)

    importance = np.zeros(X.shape[1])

    for i in range(Y.shape[1]):
        n_coefs_lars = int(np.ceil(X.shape[1] * ratio))
        if n_coefs_lars == 0: n_coefs_lars = 1 
        
        model = Lars(n_nonzero_coefs=n_coefs_lars, fit_intercept=False)
        model.fit(X, Y[:, i])
        coef = np.abs(model.coef_)
        importance = np.maximum(importance, coef) 

    importance_series = pd.Series(importance, index=feature_names)
    ranked_features = importance_series.sort_values(ascending=False)
    return ranked_features

# --- 0. Pré-requis : Définir les sous-types de référence (clusters_reference) ---
print("--- Préparation des données et définition des sous-types de référence ---")

# Chargement du fichier CSV original pour obtenir toutes les données (nécessaire pour les noms de colonnes et la structure)
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

# PCA avec 4 composantes principales (pour la référence de clustering)
pca_reference = PCA(n_components=4)
df_pca_reference = pd.DataFrame(pca_reference.fit_transform(df_normalized), columns=["PC1", "PC2", "PC3", "PC4"])

# Clustering KMeans avec 4 clusters pour définir les sous-types de référence
kmeans_reference = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_reference = kmeans_reference.fit_predict(df_pca_reference)

print(f"Sous-types de référence générés. Nombre d'échantillons: {len(clusters_reference)}")
print(f"Nombre de caractéristiques originales: {df_normalized.shape[1]}")

# --- 1. Application de UMCFS pour obtenir le classement initial ---
print("\n--- Application de l'algorithme UMCFS pour classer toutes les caractéristiques ---")

X_umcfs = df_normalized.values # UMCFS s'applique sur les données normalisées
feature_names_umcfs = df_normalized.columns.tolist()

# Utilisation des paramètres k et n_eigen par défaut ou ajustables
ranked_features_umcfs = umcfs(X_umcfs, feature_names_umcfs, k=5, n_eigen=5, ratio=1.0)

print("\nTop 10 des caractéristiques selon UMCFS (par importance) :")
print(ranked_features_umcfs.head(10).to_markdown(numalign="left", stralign="left"))


# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour UMCFS ---")

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Vous pouvez ajuster cette liste

best_ari_umcfs = -1.0
best_k_umcfs = -1
optimal_features_list_umcfs = [] # Stockera les noms des caractéristiques pour le K optimal

for k in k_values_to_test:
    # Sélectionner les top K caractéristiques basées sur le classement UMCFS
    current_selected_features_names = ranked_features_umcfs.head(k).index.tolist()

    if not current_selected_features_names:
        continue # Passer si aucune caractéristique sélectionnée pour ce K

    # Filtrer les données normalisées avec les caractéristiques sélectionnées
    X_selected_for_clustering = df_normalized[current_selected_features_names]

    # Appliquer KMeans sur ces caractéristiques sélectionnées
    kmeans_fsa_test = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters_fsa_test = kmeans_fsa_test.fit_predict(X_selected_for_clustering)

    # Calculer l'ARI
    ari = adjusted_rand_score(clusters_reference, clusters_fsa_test)

    # Mettre à jour le meilleur ARI et le K optimal
    if ari > best_ari_umcfs:
        best_ari_umcfs = ari
        best_k_umcfs = k
        optimal_features_list_umcfs = current_selected_features_names


# --- 3. Sauvegarde des résultats du UMCFS ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs d'importance dans un fichier CSV.
if optimal_features_list_umcfs:
    # Sélectionner les valeurs d'importance pour les caractéristiques optimales
    final_optimal_features_series = ranked_features_umcfs.loc[optimal_features_list_umcfs]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'UMCFS_Importance_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "umcfs_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"\nLa liste des {len(optimal_features_list_umcfs)} caractéristiques optimales (avec leurs valeurs d'importance UMCFS) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour UMCFS.")


print(f"\n--- Résultats Optimaux pour UMCFS ---")
print(f"Meilleur K pour UMCFS (selon ARI) : {best_k_umcfs}")
print(f"ARI le plus élevé : {best_ari_umcfs:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_umcfs)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_umcfs[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")






"""


import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Lars
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

# --- Fonctions internes de UMCFS ---
def construct_affinity_matrix(X, k=5, t=None):
    A = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    A = 0.5 * (A + A.T)
    D = cdist(X, X, 'euclidean')
    if t is None:
        t = np.mean(D)
    W = np.exp(-D ** 2 / (2 * t ** 2))
    W *= A.toarray()
    return W

def laplacian_eigenmap(W, n_components=5):
    L, D = laplacian(W, normed=False, return_diag=True)
    D = np.diag(D)
    eigvals, eigvecs = eigh(L, D)
    return eigvecs[:, 1:n_components+1]  # on ignore le 1er vecteur propre trivial

def umcfs(X, feature_names, k=5, n_eigen=5, ratio=1.0):
    W = construct_affinity_matrix(X, k)
    Y = laplacian_eigenmap(W, n_eigen)

    importance = np.zeros(X.shape[1])

    for i in range(Y.shape[1]):
        model = Lars(n_nonzero_coefs=int(np.ceil(X.shape[1] * ratio)))
        model.fit(X, Y[:, i])
        coef = np.abs(model.coef_)
        importance = np.maximum(importance, coef)  # max sur les fonctions propres

    importance_series = pd.Series(importance, index=feature_names)
    ranked_features = importance_series.sort_values(ascending=False)
    return ranked_features

# --- Chargement des données ---
X_normalized_path = "C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_normalized.csv"
X_df = pd.read_csv(X_normalized_path)
X = X_df.values
feature_names = X_df.columns

print("✅ Données normalisées chargées.")
print(f"Dimensions : {X.shape}")

# --- Application de UMCFS ---
print("\n--- Application de l'algorithme UMCFS (non supervisé) ---")
ranked_features_umcfs = umcfs(X, feature_names, k=5, n_eigen=5, ratio=1.0)

# --- Affichage du Top 10 ---
print("\nTop 10 des caractéristiques selon UMCFS (par importance maximale LARS) :")
print(ranked_features_umcfs.head(10).to_markdown(numalign="left", stralign="left"))

# --- Sauvegarde CSV ---
selected_features_df = ranked_features_umcfs[ranked_features_umcfs > 1e-6].to_frame(name='UMCFS_Importance')
output_csv = "selected_features_by_UMCFS.csv"
selected_features_df.to_csv(output_csv, index=True, header=True)

print(f"\nNombre de caractéristiques sélectionnées par UMCFS (importance > 1e-6) : {len(selected_features_df)}")
print(f"✅ Résultat enregistré dans : {output_csv}")

# --- Sauvegarde TXT des noms de features ---
output_txt = "selected_feature_names_UMCFS.txt"
with open(output_txt, "w") as f:
    for feature_name in selected_features_df.index:
        f.write(f"{feature_name}\n")
print(f"✅ Noms des caractéristiques enregistrés dans : {output_txt}")

"""
