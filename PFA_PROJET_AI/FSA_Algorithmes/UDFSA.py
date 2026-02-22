
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# --- Fonctions internes de UDFS (inchangées) ---
def local_dis_ana(X, knn=5):
    """
    Construit un graphe k-NN symétrique et en déduit la matrice Laplacienne.
    X: Données de forme (n_samples, n_features)
    knn: Nombre de voisins pour le graphe k-NN
    """
    # Construction d'un graphe k-NN symétrique et Laplacien
    W = kneighbors_graph(X, knn, mode='connectivity', include_self=False).toarray()
    W = np.maximum(W, W.T)  # symétriser
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

def fs_unsup_udfs(A, k, r, X0=None, max_iter=20, tol=1e-8):
    """
    Résolution itérative du problème de minimisation pour UDFS.
    A: Matrice d'entrée (A = X.T @ L @ X)
    k: Dimension de l'espace de projection / nombre de caractéristiques à sélectionner
    r: Paramètre de régularisation (gamma dans la fonction udfs)
    X0: Initialisation de la matrice de projection X
    max_iter: Nombre maximal d'itérations
    tol: Tolérance pour la convergence
    """
    m, n = A.shape # n est n_features
    if X0 is None:
        # Initialisation par défaut si X0 n'est pas fourni
        # Il est important que X0 ait les bonnes dimensions (n_features, k)
        # np.eye(n, k) crée une matrice diagonale si n=k, sinon une partie.
        # Cette initialisation peut être sensible à la convergence.
        if n < k:
            print("Warning: n_features < k in fs_unsup_udfs, adjusting k.")
            k = n
        X = np.eye(n, k) # X est (n_features, k)
    else:
        X = X0
    
    # Calcul de d (partie de la régularisation L2,1)
    Xi = np.sqrt(np.sum(X**2, axis=1) + 1e-10) # Norme L2 de chaque ligne de X
    d = 0.5 / Xi

    obj = [] # Pour suivre l'objectif
    for _ in range(max_iter):
        D = np.diag(d) # Matrice diagonale de d
        M = A + r * D
        M = (M + M.T) / 2 # S'assurer de la symétrie
        
        # Eigen-décomposition pour trouver la nouvelle X
        eigval, eigvec = eigh(M)
        idx = np.argsort(eigval) # Tri des valeurs propres par ordre croissant
        X_new = eigvec[:, idx[:k]] # Sélectionne les k vecteurs propres correspondant aux plus petites valeurs

        # Mise à jour de d
        Xi_new = np.sqrt(np.sum(X_new**2, axis=1) + 1e-10)
        d_new = 0.5 / Xi_new
        
        # Calcul de la valeur de l'objectif (pour la convergence)
        obj_val = np.trace(X_new.T @ A @ X_new) + r * np.sum(Xi_new)
        obj.append(obj_val)

        # Vérification de la convergence
        if len(obj) > 1 and abs(obj[-1] - obj[-2]) < tol:
            break
        
        X = X_new.copy()
        d = d_new.copy()

    return X, obj # X est la matrice de projection finale, obj les valeurs d'objectif

def udfs(X, n_class, knn=5, gamma=1e-5, lamda=1e-5, threshold=1e-5):
    """
    UDFS: Unsupervised Discriminative Feature Selection.
    X shape: (n_samples, n_features)
    n_class: Nombre de clusters / dimension de l'espace de projection
    knn: Paramètre k pour local_dis_ana
    gamma: Paramètre r dans fs_unsup_udfs (régularisation L2,1)
    lamda: Paramètre non utilisé dans cette implémentation fournie.
    threshold: Seuil pour la sélection initiale (sera ignoré pour l'évaluation ARI)
    """
    X = X.astype(float)
    # print("UDFS: Regularized Discriminative Feature Selection for Unsupervised Learning") # Désactivé pour moins de verbiage
    
    # Calcul du Laplacien local
    L = local_dis_ana(X, knn)
    if np.isnan(L).any():
        print("Attention: NaN dans L, cela peut indiquer un problème avec le graphe ou les données.")
        # Pour une robustesse, on peut décider de gérer cela (ex: retourner des scores de 0)
        # Ou ajuster knn, ou vérifier les données d'entrée.
        # Ici, nous allons laisser le flux de l'erreur si elle se produit et est non gérée.

    # Construire la matrice A = X.T @ L @ X
    # X est (n_samples, n_features), L est (n_samples, n_samples)
    # X.T est (n_features, n_samples)
    # A = (n_features, n_samples) @ (n_samples, n_samples) @ (n_samples, n_features)
    # A sera de forme (n_features, n_features)
    A = X.T @ L @ X
    
    # Appliquer fs_unsup_udfs pour obtenir la matrice de projection W
    W, _ = fs_unsup_udfs(A, n_class, gamma) # n_class est k dans fs_unsup_udfs

    # Les scores d'importance des caractéristiques sont la norme L2 de chaque ligne de W
    scores = np.sum(W**2, axis=1) # scores est de forme (n_features,)
    
    # ranking_idx trie les indices des caractéristiques par ordre décroissant de score
    ranking_idx = np.argsort(-scores) 

    # La sélection par seuil n'est pas utilisée pour l'évaluation ARI,
    # mais les scores complets sont retournés.
    # scores_sorted = scores[ranking_idx]
    # selected_idx = ranking_idx[scores_sorted > threshold]
    
    return ranking_idx, scores # Retourne les indices classés et tous les scores

# --- 0. Pré-requis : Définir les sous-types de référence (clusters_reference) ---
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

# PCA avec 4 composantes principales (pour la référence de clustering)
pca_reference = PCA(n_components=4)
df_pca_reference = pd.DataFrame(pca_reference.fit_transform(df_normalized), columns=["PC1", "PC2", "PC3", "PC4"])

# Clustering KMeans avec 4 clusters pour définir les sous-types de référence
kmeans_reference = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_reference = kmeans_reference.fit_predict(df_pca_reference)

print(f"Sous-types de référence générés. Nombre d'échantillons: {len(clusters_reference)}")
print(f"Nombre de caractéristiques originales: {df_normalized.shape[1]}")


# --- 1. Application de UDFS pour obtenir le classement initial ---
print("\n--- Application de l'algorithme UDFS pour classer toutes les caractéristiques ---")

X_udfs_input = df_normalized.values # UDFS attend (n_samples, n_features)
feature_names_udfs = df_normalized.columns.tolist()

# n_class pour UDFS est la dimension de l'espace de projection ou le nombre de clusters
num_clusters = len(np.unique(clusters_reference)) # Devrait être 4

# Appliquer UDFS
ranking_idx_udfs, scores_udfs = udfs(X_udfs_input, n_class=num_clusters, knn=5, gamma=1e-5, lamda=1e-5, threshold=1e-5)

# Créer un Series pandas avec les scores triés pour l'évaluation ARI
ranked_features_udfs = pd.Series(scores_udfs, index=feature_names_udfs).sort_values(ascending=False)

print("\nTop 10 des caractéristiques selon UDFS (par importance L2-norme des coefficients) :")
print(ranked_features_udfs.head(10).to_markdown(numalign="left", stralign="left"))


# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour UDFS ---")

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Ajustez cette liste si nécessaire

best_ari_udfs = -1.0
best_k_udfs = -1
optimal_features_list_udfs = [] # Stockera les noms des caractéristiques pour le K optimal

for k in k_values_to_test:
    # Sélectionner les top K caractéristiques basées sur le classement UDFS
    current_selected_features_names = ranked_features_udfs.head(k).index.tolist()

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
    if ari > best_ari_udfs:
        best_ari_udfs = ari
        best_k_udfs = k
        optimal_features_list_udfs = current_selected_features_names


# --- 3. Sauvegarde des résultats du UDFS ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs d'importance dans un fichier CSV.
if optimal_features_list_udfs:
    # Sélectionner les valeurs d'importance pour les caractéristiques optimales
    final_optimal_features_series = ranked_features_udfs.loc[optimal_features_list_udfs]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'UDFS_Importance_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "udfs_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"\nLa liste des {len(optimal_features_list_udfs)} caractéristiques optimales (avec leurs valeurs d'importance UDFS) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour UDFS.")


print(f"\n--- Résultats Optimaux pour UDFS ---")
print(f"Meilleur K pour UDFS (selon ARI) : {best_k_udfs}")
print(f"ARI le plus élevé : {best_ari_udfs:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_udfs)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_udfs[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")


"""
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh

def local_dis_ana(X, knn=5):
    # Construction d'un graphe k-NN symétrique et Laplacien
    W = kneighbors_graph(X, knn, mode='connectivity', include_self=False).toarray()
    W = np.maximum(W, W.T)  # symétriser
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

def fs_unsup_udfs(A, k, r, X0=None, max_iter=20, tol=1e-8):
    # Résolution de min_{X'X=I} Tr(X'AX) + r*||X||_21
    m, n = A.shape
    if X0 is None:
        d = np.ones(n)
        X = np.eye(n, k)
    else:
        X = X0
        Xi = np.sqrt(np.sum(X**2, axis=1) + 1e-10)
        d = 0.5 / Xi

    obj = []
    for _ in range(max_iter):
        D = np.diag(d)
        M = A + r * D
        M = (M + M.T) / 2
        eigval, eigvec = eigh(M)
        idx = np.argsort(eigval)
        X = eigvec[:, idx[:k]]
        Xi = np.sqrt(np.sum(X**2, axis=1) + 1e-10)
        d = 0.5 / Xi
        obj_val = np.trace(X.T @ A @ X) + r * np.sum(Xi)
        obj.append(obj_val)
        if len(obj) > 1 and abs(obj[-1] - obj[-2]) < tol:
            break
    return X, obj

def udfs(X, n_class, knn=5, gamma=1e-5, lamda=1e-5, threshold=1e-5):
    # X shape: (n_samples, n_features)
    X = X.astype(float)
    print("UDFS: Regularized Discriminative Feature Selection for Unsupervised Learning")
    
    # Calcul du Laplacien local
    L = local_dis_ana(X, knn)
    if np.isnan(L).any():
        print("Attention: NaN dans L, renvoi des indices naturels")
        return np.arange(X.shape[1])

    A = X.T @ L @ X
    W, _ = fs_unsup_udfs(A, n_class, gamma)

    scores = np.sum(W**2, axis=1)
    ranking_idx = np.argsort(-scores)  # décroissant

    # Application seuil sur score
    scores_sorted = scores[ranking_idx]
    selected_idx = ranking_idx[scores_sorted > threshold]

    return ranking_idx, selected_idx, scores

# === Exemple d'utilisation ===

# Chargement données normalisées
X_df = pd.read_csv("C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_normalized.csv")
X_np = X_df.values  # (n_samples, n_features)
feature_names = X_df.columns

# Application UDFS
ranking_all, selected_idx, scores = udfs(X_np, n_class=2, knn=5, gamma=1e-5, lamda=1e-5, threshold=1e-5)

# Affichage des top features
print("\nTop 10 des caractéristiques UDFS (avec seuil > 1e-5) :")
top_features = feature_names[selected_idx[:10]]
top_scores = scores[selected_idx[:10]]
for f, s in zip(top_features, top_scores):
    print(f"{f} : {s:.6e}")

# Sauvegarde des résultats
pd.Series(scores, index=feature_names).to_csv("udfs_scores_all.csv")
pd.Series(feature_names[selected_idx]).to_csv("udfs_selected_features.txt", index=False)

"""