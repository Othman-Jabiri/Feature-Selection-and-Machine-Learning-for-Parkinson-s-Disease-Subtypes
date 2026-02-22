
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import os

# --- Fonctions internes de UFSOL (Adaptées) ---
def triplet_laplacian(X, k=5):
    """
    Calcule le Laplacien basé sur un graphe kNN.
    X doit être de forme (n_samples, n_features) pour euclidean_distances.
    """
    n = X.shape[0]
    D = euclidean_distances(X)
    np.fill_diagonal(D, np.inf) # Éviter le point lui-même
    idx = np.argsort(D, axis=1)[:, :k] # Indices des k plus proches voisins
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i]] = 1
    W = np.maximum(W, W.T) # Rendre symétrique
    Dg = np.diag(W.sum(axis=1)) # Matrice de degré
    return Dg - W # Laplacien non normalisé

def ufsol(X, n_class, alpha, beta, p0='sample', k_neighbors=5, max_iter=10, tol=1e-7):
    """
    Implémentation de l'algorithme UFSOL.
    X: Données d'entrée de forme (n_features, n_samples)
    n_class: Nombre de clusters / dimension de l'espace latent
    alpha, beta: Paramètres de régularisation
    p0: 'sample' pour Laplacien sur les échantillons, 'feature' pour Laplacien sur les caractéristiques
    k_neighbors: Nombre de voisins pour la construction du Laplacien
    max_iter: Nombre maximal d'itérations
    tol: Tolérance pour la convergence
    """
    n_fea, n_smp = X.shape

    # Construction du Laplacien L
    if p0 == 'feature':
        # Laplacien sur le graphe de caractéristiques. X.T est (n_samples, n_features)
        L = triplet_laplacian(X.T, k_neighbors)
    else: # p0 == 'sample'
        # Laplacien sur le graphe d'échantillons (L0), puis transformé en espace de caractéristiques
        L0 = triplet_laplacian(X.T, k_neighbors) # X.T est (n_samples, n_features)
        L = X @ L0 @ X.T # L sera (n_features, n_features)

    # Initialisation de W
    # W représente les coefficients de sélection de caractéristiques (projection)
    W = np.eye(n_fea, n_class) # Initialisation à la matrice identité (ou aléatoire)
    
    # Initialisation de V (embeddings de l'espace échantillon)
    # kmeans_transform est appliqué sur (W.T @ X).T qui est (n_samples, n_class)
    # Le résultat est (n_samples, n_class)
    # Puis, une transposition finale pour que V soit (n_class, n_samples)
    km = KMeans(n_clusters=n_class, n_init=10, random_state=42)
    # (W.T @ X).T est de forme (n_samples, n_class)
    V_kmeans = km.fit_transform((W.T @ X).T)
    V = V_kmeans.T # V sera (n_class, n_samples)

    # U (non explicitement utilisé pour le calcul final de scores) est souvent une matrice d'orthonormalisation
    U = None
    W_old = W.copy()

    for it in range(max_iter):
        # print(f"[UFSOL] Iteration {it + 1}/{max_iter}...")

        # Mise à jour de R (matrice diagonale pour la régularisation L2,1)
        rdiag_values = 2 * np.sqrt((W ** 2).sum(axis=1)) # Somme des carrés par ligne (caractéristique)
        R = np.diag(1 / np.maximum(rdiag_values, tol)) # Empêche la division par zéro

        # Mise à jour de U, V (embeddings)
        # Ceci est une approximation QR. Le V ici est une 'approximation'
        # de l'espace des clusters, mais le V provenant de KMeans initial est souvent utilisé
        # ou mis à jour de manière itérative avec V_new = W.T @ X.
        # Pour rester fidèle à la structure, nous utilisons la mise à jour via QR
        # (W.T @ X) est (n_class, n_samples)
        Q, R_qr = np.linalg.qr(W.T @ X) # Q est (n_class, n_class), R_qr est (n_class, n_samples)
        U = Q # Q est orthogonal
        V = R_qr # V est triangulaire supérieure (ici, partie de la décomposition QR)

        # Mise à jour de W (coefficients de sélection de caractéristiques)
        # T est la matrice utilisée pour l'Eigen-décomposition
        # X est (n_fea, n_smp)
        # V est (n_class, n_smp)
        # L est (n_fea, n_fea)
        
        # Terme X @ (V.T @ V) @ X.T est de forme (n_fea, n_fea)
        # (V.T @ V) est (n_smp, n_smp)
        # X @ (V.T @ V) @ X.T -> (n_fea, n_smp) @ (n_smp, n_smp) @ (n_smp, n_fea) = (n_fea, n_fea)
        
        T = beta * R + X @ X.T - X @ (V.T @ V) @ X.T + alpha * L
        T = (T + T.T) / 2 # S'assurer que T est symétrique

        try:
            # Calcul des vecteurs propres pour W
            eigv, eigvec = eigh(T)
            # Sélectionne les n_class premiers vecteurs propres correspondant aux plus petites valeurs propres
            # Ces vecteurs forment la matrice W.
            W = eigvec[:, :n_class]
        except Exception as e:
            print(f"Erreur lors du calcul des vecteurs propres pour W : {e}")
            break # Sortir de la boucle d'itération en cas d'échec

        # Vérification de la convergence
        if it > 0 and np.linalg.norm(W - W_old, 'fro') < tol: # Frobenius norm pour la convergence
            # print(f"Converged at iteration {it + 1}.")
            break
        W_old = W.copy()

    # Les scores de caractéristiques sont la norme L2 de chaque ligne de W
    # Une ligne de W correspond à une caractéristique.
    scores = np.sqrt(np.sum(W**2, axis=1)) 
    return scores # Retourne seulement les scores pour l'évaluation ARI

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

# --- 1. Application de UFSOL pour obtenir le classement initial ---
print("\n--- Application de l'algorithme UFSOL pour classer toutes les caractéristiques ---")

X_ufsol_input = df_normalized.values.T # UFSOL attend (features, samples)
feature_names_ufsol = df_normalized.columns.tolist()

# n_class est le nombre de clusters / dimension de l'espace latent (aligné avec la référence)
num_clusters = len(np.unique(clusters_reference)) # Devrait être 4

# Appliquer UFSOL
scores_ufsol_all = ufsol(X_ufsol_input, n_class=num_clusters, alpha=1, beta=100, p0='sample', k_neighbors=5, max_iter=10)

# Création du DataFrame pour le classement complet
ranked_features_ufsol = pd.Series(scores_ufsol_all, index=feature_names_ufsol).sort_values(ascending=False)

print("\nTop 10 des caractéristiques selon UFSOL (par importance L2-norme des coefficients) :")
print(ranked_features_ufsol.head(10).to_markdown(numalign="left", stralign="left"))


# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour UFSOL ---")

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Ajustez cette liste si nécessaire

best_ari_ufsol = -1.0
best_k_ufsol = -1
optimal_features_list_ufsol = [] # Stockera les noms des caractéristiques pour le K optimal

for k in k_values_to_test:
    # Sélectionner les top K caractéristiques basées sur le classement UFSOL
    current_selected_features_names = ranked_features_ufsol.head(k).index.tolist()

    if not current_selected_features_names:
        continue # Passer si aucune caractéristique sélectionnée pour ce K

    # Filtrer les données normalisées (qui sont samples x features) avec les caractéristiques sélectionnées
    X_selected_for_clustering = df_normalized[current_selected_features_names]

    # Appliquer KMeans sur ces caractéristiques sélectionnées
    kmeans_fsa_test = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters_fsa_test = kmeans_fsa_test.fit_predict(X_selected_for_clustering)

    # Calculer l'ARI
    ari = adjusted_rand_score(clusters_reference, clusters_fsa_test)

    # Mettre à jour le meilleur ARI et le K optimal
    if ari > best_ari_ufsol:
        best_ari_ufsol = ari
        best_k_ufsol = k
        optimal_features_list_ufsol = current_selected_features_names


# --- 3. Sauvegarde des résultats du UFSOL ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs d'importance dans un fichier CSV.
if optimal_features_list_ufsol:
    # Sélectionner les valeurs d'importance pour les caractéristiques optimales
    final_optimal_features_series = ranked_features_ufsol.loc[optimal_features_list_ufsol]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'UFSOL_Importance_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "ufsol_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"\nLa liste des {len(optimal_features_list_ufsol)} caractéristiques optimales (avec leurs valeurs d'importance UFSOL) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour UFSOL.")


print(f"\n--- Résultats Optimaux pour UFSOL ---")
print(f"Meilleur K pour UFSOL (selon ARI) : {best_k_ufsol}")
print(f"ARI le plus élevé : {best_ari_ufsol:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_ufsol)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_ufsol[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")





"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh

def triplet_laplacian(X, k=5):
    # Approximatif : graphe kNN + pondération
    n = X.shape[0]
    D = euclidean_distances(X)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idx[i]] = 1
    W = np.maximum(W, W.T)
    Dg = np.diag(W.sum(1))
    return Dg - W

def ufsol(X, n_class, alpha, beta, p0='sample', k=5, max_iter=10, tol=1e-7):
    # X: (features, samples)
    n_fea, n_smp = X.shape

    if p0 == 'feature':
        L = triplet_laplacian(X.T, k)
    else:
        L0 = triplet_laplacian(X.T, k)
        L = X @ L0 @ X.T

    W = np.eye(n_fea, n_class)
    km = KMeans(n_clusters=n_class, n_init=10)
    V = km.fit_transform((W.T @ X).T).T
    U = None

    for it in range(max_iter):
        # R update
        rdiag = 2 * np.sqrt((W ** 2).sum(1))
        R = np.diag(1 / np.maximum(rdiag, tol))

        # Update U, V by solving U*V ≈ W^T X
        U, Vt = np.linalg.qr(W.T @ X)  # approx. orthog.
        V = Vt[:, :n_smp]

        # W update
        T = beta*R + X @ X.T - X @ (V.T @ V) @ X.T + alpha*L
        T = (T + T.T) / 2
        eigv, eigvec = eigh(T)
        W = eigvec[:, :n_class]

        # Stopping
        if it and np.linalg.norm(W - W_old) < tol:
            break
        W_old = W.copy()

    scores = np.sum(W**2, axis=1)
    return W, scores

# === Pipeline UFSOL avec seuil ===

# 1. Chargement des données
X_df = pd.read_csv("C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_normalized.csv")
X = X_df.values.T  # transpose: features × samples
names = X_df.columns

# 2. Application
W, scores = ufsol(X, n_class=2, alpha=1, beta=100, p0='sample', k=5, max_iter=10)

# 3. Tri et seuil
ser = pd.Series(scores, index=names).sort_values(ascending=False)
threshold = 1e-5
sel = ser[ser > threshold]

print("Top 10 features UFSOL (>1e-5):")
print(sel.head(10).to_markdown())

# 4. Sauvegardes
ser.to_frame("score").to_csv("ufscores_all.csv")
sel.index.to_series(name="feature").to_csv("ufs_selected.txt", index=False)
"""