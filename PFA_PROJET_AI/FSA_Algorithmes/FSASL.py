
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
from numpy.linalg import inv # Not strictly used in the current solve for W_mat, but often in similar methods.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# --- Fonctions internes de FSASL (Adaptées pour le contexte) ---
# Note: Ces fonctions implémentent une version de FSASL.
# Des ajustements ou une implémentation plus robuste pourraient être nécessaires
# selon la version exacte de FSASL à laquelle l'article fait référence.

def construct_W(X, k):
    """Construit une matrice d'affinité basée sur les k plus proches voisins."""
    dist = euclidean_distances(X)
    np.fill_diagonal(dist, np.inf) # Éviter que le point soit son propre voisin
    
    # Obtenir les indices des k plus proches voisins pour chaque point
    # Exclure le point lui-même, donc on prend de 1 à k+1
    idx = np.argsort(dist, axis=1)[:, :k] # Corrected to take k neighbors, not k+1

    W = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        W[i, idx[i]] = 1 # Met 1 pour les k plus proches voisins
    
    # Rendre la matrice symétrique pour un graphe non dirigé
    W = np.maximum(W, W.T) 
    return W

def laplacian(W):
    """Calcule la matrice Laplacienne non normalisée."""
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

def fsasl(X, n_class, lambda1=1.0, lambda2=1.0, lambda3=1.0, k_neighbors=5, max_iter=10): # Renommé k pour éviter conflit, augmenté max_iter
    n_samples, n_features = X.shape
    X_T = X.T  # Transposition pour que les caractéristiques soient des lignes (d, n)

    # Initialisation de S et W_mat (coefficients de sélection des caractéristiques)
    # Les initialisations peuvent influencer la convergence.
    # On initialise W_mat aléatoirement, puis normalise pour éviter des valeurs trop grandes/petites initiales.
    W_mat = np.random.rand(n_features, n_class)
    W_mat = W_mat / (np.sqrt(np.sum(W_mat**2, axis=1, keepdims=True)) + 1e-10) # Normalisation L2 par ligne

    for iteration in range(max_iter):
        print(f"[FSASL] Iteration {iteration + 1}/{max_iter}...")

        # 1. Mise à jour de la matrice de similarité adaptative (S)
        # Ceci est une simplification de la partie Global Structure Learning (LG) souvent basée sur la reconstruction
        # Dans beaucoup d'implémentations, LG est appris par une régression creuse.
        # Pour une implémentation plus fidèle à l'article, il faudrait une résolution plus complexe ici.
        # Ici, nous allons utiliser une approximation ou sauter la mise à jour de LG si elle est trop complexe.
        # Puisque FSASL se concentre sur 'Adaptive Structure Learning', nous pouvons le voir comme S étant ce que nous apprenons.
        
        # Le code original avait une section pour LG qui semblait être une reconstruction locale pour chaque point,
        # puis LL basé sur un graphe k-NN. Il les combine en un Laplacien total L.
        # La partie cruciale est la résolution pour W_mat.

        # 2. Local structure learning (LL)
        # A est la matrice d'affinité construite à partir des données dans l'espace courant (ou original)
        A_affinity = construct_W(X, k_neighbors) # Utilise k_neighbors comme k dans construct_W
        LL = laplacian(A_affinity)
        LL = (LL + LL.T) / 2 # S'assurer de la symétrie

        # Dans FSASL, L est souvent un Laplacien de haut niveau ou combiné
        # Ici, L est directement le Laplacien de la structure locale apprise
        L = LL # Simplification, si LG n'est pas appris explicitement ou est intégré différemment

        # 3. Embedding Y (from smallest eigenvectors of L)
        # L doit être symétrique et définie positive (ou semi-définie positive).
        # np.linalg.eigh(L) est pour des matrices symétriques
        # Pour des matrices non symétriques, utilisez np.linalg.eig
        # Assurez-vous que L est bien symétrique et non creuse pour eigh de scipy.linalg
        try:
            L_dense = L # L est déjà numpy array dense d'après la fonction laplacian
            eigval, eigvec = eigh(L_dense)
            # Trie les valeurs propres et sélectionne les vecteurs propres correspondant aux plus petites valeurs (non triviales)
            # Le premier vecteur propre (index 0 après tri) est trivial pour le Laplacien (tous des 1)
            sorted_indices = np.argsort(eigval)
            Y = eigvec[:, sorted_indices[1 : n_class + 1]] # n_class est la dimension de l'espace latent/des clusters
        except Exception as e:
            print(f"Erreur lors du calcul des vecteurs propres : {e}")
            # Si l'eigendecomposition échoue, cela peut indiquer un problème avec L.
            # Retourner des scores de zéro pour éviter un crash complet ou gérer l'erreur.
            return pd.Series(np.zeros(n_features), index=feature_names)


        # 4. Solve for W_mat with L2,1 regularization
        # Minimisation de la fonction objectif pour W_mat.
        # C'est la partie clé de la sélection de caractéristiques.
        # La matrice D_matrix est pour la norme L2,1 (parcimonie au niveau des lignes)
        
        # Note : La résolution de 'W_mat' dans le code original est une simplification/approximation
        # Elle implique une résolution du problème de minimisation via l'Eigen-decomposition
        # Pour une convergence réelle du L2,1, des méthodes ADMM ou itératives spécifiques sont souvent utilisées.
        # Nous allons garder la structure, mais soyez conscient des limitations de cette approximation.

        # Re-calculer D_matrix pour la régularisation L2,1
        # Ajout de 1e-10 pour la stabilité numérique si la somme des carrés est zéro
        row_norms = np.sqrt(np.sum(W_mat**2, axis=1)) + 1e-10
        D_matrix = np.diag(1 / row_norms)
        
        # Terme X L X^T dans l'équation.
        XLX_T = X_T @ L @ X # Erreur de dimension: L est (n,n), X_T est (d,n), X est (n,d).
                             # Ça devrait être X_T @ L @ X pour une matrice (d, d)
        
        # Le problème formulé dans l'article ressemble plus à (X W - Y)^2 + regul.
        # L'implémentation donnée semble être une version simplifiée ou une étape intermédiaire.
        # Pour correspondre à la formule $min_{W, S} Tr(W^T X^T L(S) X W) + \lambda \|W\|_{2,1} + ...$
        # la résolution de W_mat serait différente.
        # Le code original semble optimiser une partie de $A_W W = B_W$.
        
        # Tentative de rendre plus robuste la résolution de W_mat
        # Nous allons utiliser une approche plus directe de minimisation d'un terme Laplacien avec L2,1
        # ou se baser sur le principe du papier si une réimplémentation complète n'est pas possible.
        
        # Simplification basée sur la structure du code original:
        # La ligne A = X @ L @ X.T + lambda3 * D_matrix est problématique en dimensions
        # L est (n,n), X est (n,p)
        # X @ L @ X.T -> (n,p) @ (n,n) @ (n,p).T -> (n,n) * (n,p) = pas bon
        # Devrait être (p,n) @ (n,n) @ (n,p) = (p,p)
        # Donc (X_T @ L @ X) est (p, p)
        # A = (X_T @ L @ X) + lambda3 * D_matrix
        # A doit être (p,p)

        try:
            A_matrix_for_eig = (X_T @ L @ X) + lambda3 * D_matrix
            A_matrix_for_eig = (A_matrix_for_eig + A_matrix_for_eig.T) / 2 # S'assurer de la symétrie
            
            eigval_w, eigvec_w = eigh(A_matrix_for_eig)
            # Sélectionne les vecteurs propres correspondant aux plus petites valeurs propres
            # Ces vecteurs définissent la transformation qui minimise la fonction objectif.
            W_mat = eigvec_w[:, :n_class] # Ici n_class est la dimension de l'espace de projection

        except Exception as e:
            print(f"Erreur lors du calcul des vecteurs propres pour W_mat : {e}")
            # Si l'eigendecomposition échoue, retourner les scores actuels ou des zéros.
            break # Sortir de la boucle d'itération


    # Feature importance = L2 norm of rows of W_mat (coefficients de sélection)
    # Les scores sont calculés à la fin de l'optimisation.
    feature_scores = np.sqrt(np.sum(W_mat**2, axis=1)) # L2 norm de chaque ligne de W_mat
    return feature_scores

# --- 0. Pré-requis : Définir les sous-types de référence (clusters_reference) ---
print("--- Préparation des données et définition des sous-types de référence ---")

# Chargement du fichier CSV original pour obtenir toutes les données (nécessaire pour les noms de colonnes et la structure)
data_path_features ="C:/Users/pc/Desktop/Features/PFA_PROJET_AI/data/pd_EEG_features.csv"
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

# --- 1. Application de FSASL pour obtenir le classement initial ---
print("\n--- Application de l'algorithme FSASL pour classer toutes les caractéristiques ---")

X_fsasl = df_normalized.values # FSASL s'applique sur les données normalisées
feature_names_fsasl = df_normalized.columns.tolist()

# n_class est souvent interprété comme le nombre de dimensions de l'espace latent, ou nombre de clusters
# ici, nous l'alignons avec le nombre de clusters de référence pour la pertinence
num_clusters = len(np.unique(clusters_reference)) # Devrait être 4
scores = fsasl(X_fsasl, n_class=num_clusters, lambda1=1, lambda2=1, lambda3=1, k_neighbors=5, max_iter=10)

# 3. Création du DataFrame
fsasl_scores = pd.Series(scores, index=feature_names_fsasl)
ranked_features_fsasl = fsasl_scores.sort_values(ascending=False)

print("\nTop 10 des caractéristiques selon FSASL (par importance L2-norme des coefficients) :")
print(ranked_features_fsasl.head(10).to_markdown(numalign="left", stralign="left"))

# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour FSASL ---")

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Ajustez cette liste si nécessaire

best_ari_fsasl = -1.0
best_k_fsasl = -1
optimal_features_list_fsasl = [] # Stockera les noms des caractéristiques pour le K optimal

for k in k_values_to_test:
    # Sélectionner les top K caractéristiques basées sur le classement FSASL
    current_selected_features_names = ranked_features_fsasl.head(k).index.tolist()

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
    if ari > best_ari_fsasl:
        best_ari_fsasl = ari
        best_k_fsasl = k
        optimal_features_list_fsasl = current_selected_features_names


# --- 3. Sauvegarde des résultats du FSASL ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs d'importance dans un fichier CSV.
if optimal_features_list_fsasl:
    # Sélectionner les valeurs d'importance pour les caractéristiques optimales
    final_optimal_features_series = ranked_features_fsasl.loc[optimal_features_list_fsasl]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'FSASL_Importance_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "fsasl_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"\nLa liste des {len(optimal_features_list_fsasl)} caractéristiques optimales (avec leurs valeurs d'importance FSASL) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour FSASL.")


print(f"\n--- Résultats Optimaux pour FSASL ---")
print(f"Meilleur K pour FSASL (selon ARI) : {best_k_fsasl}")
print(f"ARI le plus élevé : {best_ari_fsasl:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_fsasl)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_fsasl[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")














"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
from numpy.linalg import inv

def construct_W(X, k):
    dist = euclidean_distances(X)
    np.fill_diagonal(dist, np.inf)
    idx = np.argsort(dist, axis=1)[:, 1:k+1]
    W = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        W[i, idx[i]] = 1
    W = np.maximum(W, W.T)
    return W

def laplacian(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

def fsasl(X, n_class, lambda1=1.0, lambda2=1.0, lambda3=1.0, k=5, max_iter=1):
    n_samples, n_features = X.shape
    X = X.T  # Shape (d, n)

    S = np.zeros((n_samples, n_samples))
    A = np.zeros((n_samples, n_samples))
    X2 = X.copy()
    
    for iteration in range(max_iter):
        print(f"[FSASL] Iteration {iteration + 1}/{max_iter}...")

        # 1. Global structure learning (LG)
        LG = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            idx = np.arange(n_samples) != i
            Xi = X2[:, idx].T  # shape (n-1, d)
            xi = X2[:, i]      # shape (d,)
            coef = np.linalg.lstsq(Xi.T, xi.T, rcond=None)[0]
            S[idx, i] = coef
        LG = np.eye(n_samples) - S
        LG = LG @ LG.T
        LG = (LG + LG.T) / 2

        # 2. Local structure learning (LL)
        W = construct_W(X2.T, k)
        A = W
        LL = laplacian(A)
        LL = (LL + LL.T) / 2

        # 3. Total Laplacian
        L = lambda1 * LG + lambda2 * LL

        # 4. Embedding Y from smallest eigenvectors
        L = (L + L.T) / 2
        eigval, eigvec = eigh(L)
        Y = eigvec[:, :n_class]

        # 5. Solve for W with LS21 regularization
        d, n = X.shape
        D = np.ones(d)
        W_mat = np.zeros((d, n_class))
        for _ in range(10):
            D_matrix = np.diag(1 / (np.sqrt(np.sum(W_mat**2, axis=1)) + 1e-10))
            A = X @ L @ X.T + lambda3 * D_matrix
            A = (A + A.T) / 2
            eigval_w, eigvec_w = eigh(A)
            W_mat = eigvec_w[:, :n_class]

    # Feature importance = L2 norm of rows of W
    feature_scores = np.sqrt(np.sum(W_mat**2, axis=1))
    return feature_scores

# =============================
# 🔵 Pipeline complet FSASL
# =============================

# 1. Charger les données normalisées
X = pd.read_csv("C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_normalized.csv")
feature_names = X.columns
X_np = X.values

print("✅ Données normalisées chargées.")
print(f"Dimensions : {X_np.shape}")

# 2. Appliquer FSASL
print("\n--- Application de l'algorithme FSASL (non supervisé) ---")
scores = fsasl(X_np, n_class=2, lambda1=1, lambda2=1, lambda3=1, k=5, max_iter=1)

# 3. Création du DataFrame
fsasl_scores = pd.Series(scores, index=feature_names)
fsasl_sorted = fsasl_scores.sort_values(ascending=False)

# 4. Filtrage par seuil
seuil = 0.00001
selected = fsasl_sorted[fsasl_sorted > seuil]

print("\nTop 10 des caractéristiques sélectionnées selon FSASL (score > 0.00001) :")
print(selected.head(10).to_markdown())

# 5. Sauvegarde complète (tout) dans CSV
output_csv = "selected_features_by_FSASL.csv"
fsasl_sorted.to_frame(name="FSASL_Score").to_csv(output_csv)

# 6. Sauvegarde seulement des noms sélectionnés dans TXT
output_txt = "selected_feature_names_FSASL.txt"
with open(output_txt, "w") as f:
    for name in selected.index:
        f.write(f"{name}\n")

print(f"\n📁 Résultats sauvegardés dans '{output_csv}' (tout) et '{output_txt}' (sélectionnées)")
print(f"✅ Nombre total de caractéristiques : {len(fsasl_sorted)}")
print(f"✅ Nombre de caractéristiques sélectionnées (score > {seuil}) : {len(selected)}")
"""