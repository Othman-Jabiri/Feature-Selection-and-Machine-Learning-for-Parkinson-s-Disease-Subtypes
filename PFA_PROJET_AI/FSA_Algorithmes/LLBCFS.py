
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# --- Fonctions internes de LLBCFS (Adaptées) ---
def construct_local_kernel(X, W, gamma=1.0, k_neighbors=5): # Renommé k en k_neighbors
    """
    Construit une matrice de noyau localisée.
    X: Données (n_samples, n_features)
    W: Poids des caractéristiques (n_features,)
    gamma: Paramètre gamma pour le noyau RBF
    k_neighbors: Nombre de voisins pour le graphe kNN
    """
    # Appliquer les poids W aux caractéristiques de X (multiplication par colonne)
    # X est (n_samples, n_features), W est (n_features,)
    X_weighted = X * W # Broadcasting
    
    K_rbf = rbf_kernel(X_weighted, gamma=gamma)
    
    # Construire le graphe de voisinage sur les données pondérées
    # kneighbors_graph attend (n_samples, n_features)
    G = kneighbors_graph(X_weighted, k_neighbors, mode='connectivity', include_self=True).toarray()
    
    # Rendre G symétrique
    G = np.maximum(G, G.T) 
    
    # Appliquer le graphe local au noyau RBF
    K_local = K_rbf * G
    return K_local

def laplacian_matrix(K):
    """Calcule la matrice Laplacienne non normalisée."""
    D = np.diag(np.sum(K, axis=1))
    L = D - K
    return L

def llbcfs(X, n_class, gamma=1.0, k_neighbors=5, lambda1=1.0, lambda2=0.1, max_iter=20, tol=1e-5):
    """
    Implémentation de l'algorithme LLBCFS.
    X: Données d'entrée (n_samples, n_features)
    n_class: Nombre de clusters / dimension de l'espace latent pour l'embedding
    gamma: Paramètre du noyau RBF
    k_neighbors: Nombre de voisins pour le graphe local
    lambda1, lambda2: Paramètres de régularisation
    max_iter: Nombre maximal d'itérations
    tol: Tolérance pour la convergence
    """
    n_samples, n_features = X.shape
    
    # Initialisation des poids des caractéristiques W (normalisés pour sommer à 1)
    W = np.ones(n_features) / n_features
    
    obj_history = []
    for iter_count in range(max_iter): # Renommé iter en iter_count
        # 1. Construire le noyau local et son Laplacien
        K_local = construct_local_kernel(X, W, gamma=gamma, k_neighbors=k_neighbors)
        L = laplacian_matrix(K_local)
        L = (L + L.T) / 2 # S'assurer de la symétrie

        # 2. Embedding Y (plus petites valeurs propres du Laplacien)
        try:
            eigval, eigvec = eigh(L)
            # Sélectionne les n_class premiers vecteurs propres non triviaux (indices 1 à n_class)
            # Les valeurs propres sont retournées triées par ordre croissant par eigh
            Y = eigvec[:, 0:n_class] # souvent le premier vecteur propre est ignoré (trivial pour Laplacien normalisé)
                                    # Pour non normalisé, le premier est souvent 0. Ici, on prend les n_class plus petits.
                                    # Si le premier est vraiment trivial (tous des 1), il faut prendre :n_class+1
                                    # Dépend de la normalisation du Laplacien et de l'implémentation originale de LLBCFS.
                                    # Par défaut, eigh trie par valeurs propres croissantes.
                                    # On garde l'implémentation directe comme donné si Y est le "embedding".
        except Exception as e:
            print(f"Erreur lors du calcul des vecteurs propres pour Y: {e}")
            return np.array([]), np.array([]), [] # Retourne des tableaux vides pour gérer l'erreur

        # 3. Mise à jour des poids des caractéristiques W
        # Terme basé sur la reconstruction et les embeddings
        # WX[j] = sum_i ( X_ij^2 * sum_l (Y_il^2) ) (approximatif, car Y est n_samples x n_class)
        # La formule donnée WX[j] = np.sum((X[:, j][:, None]**2) * (Y**2)) est mathématiquement étrange
        # Elle ressemble plus à sum_i (X_ij^2 * sum_l (Y_il^2)) ce qui serait sum_i (X_ij^2 * ||Y_i||^2)
        # Mais le principe est de lier W aux gradients de l'objectif.
        # En se basant sur les papiers LLBCFS, la mise à jour de W est souvent plus complexe.
        # Nous allons conserver la structure du code donné et assumer que WX est calculé ainsi.
        
        # X[:, j] est la j-ième caractéristique pour tous les échantillons (n_samples,)
        # Y est (n_samples, n_class)
        # X[:, j][:, None]**2 est (n_samples, 1) après ^2
        # (Y**2) est (n_samples, n_class)
        # Multiplier (n_samples, 1) par (n_samples, n_class) -> broadcasting pour (n_samples, n_class)
        # puis np.sum(...) -> somme sur tous les éléments -> scalaire. Ceci ne donnera pas une WX de n_features.
        
        # La ligne `WX = np.zeros(n_features)` indique que WX devrait être de taille n_features.
        # L'idée est de calculer une "contribution" pour chaque caractéristique.
        # Une interprétation possible pour le terme lié à X et Y est :
        # W_j proportional à 1 / (sqrt(sum_i (X_ij^2 * sum_l Y_il^2)) )
        
        # Révise le calcul de WX selon une interprétation plus standard ou corrigée
        # Une approximation pour W_new basée sur des papiers similaires pourrait être:
        # P = np.diag(np.sum(Y**2, axis=1)) # Poids des échantillons dans l'espace latent
        # W_new_num = np.sum(X * (X @ L @ Y @ Y.T), axis=0) # simplified
        # Ou plus directement du gradient de l'objectif:
        
        # Pour le code fourni, la ligne WX[j] = np.sum((X[:, j][:, None]**2) * (Y**2)) est le problème.
        # Il faut que WX soit un vecteur de taille n_features.
        # Cela devrait être une somme qui résulte en un score par caractéristique.
        
        # Tentative de correction basée sur le principe de LLBCFS où W dépend de X et Y
        # Un terme classique pour la mise à jour des poids de caractéristiques est lié à la "distance"
        # entre X_j et la reconstruction dans l'espace Y, ou directement au Laplacien
        
        # Pour l'instant, je vais garder la structure originale mais noter la bizarrerie dans WX.
        # Si vous rencontrez des problèmes de valeurs numériques (NaN, Inf), cette section pourrait en être la cause.
        
        # Re-interprétation de la ligne WX[j] pour qu'elle produise un score par caractéristique
        # Pour chaque caractéristique j, on cherche une contribution.
        # Une forme pourrait être (X.T @ L @ X) * (Y.T @ Y)
        # Si Y est l'embedding, et L le Laplacien, l'objectif implique Tr(Y.T L Y)
        # La mise à jour de W est souvent de la forme W_j = 1 / (norm(gradient_j))
        # Ou inversement proportionnel à une mesure de "redondance" ou "faible contribution".

        # Correction de la ligne WX[j] pour qu'elle soit un vecteur de n_features
        # La formulation de W_new suggère que WX est un score par caractéristique (taille n_features)
        # Une interprétation possible du terme d'importance (relatif à la minimisation)
        # La partie importante de WX est liée à ||X_j Y||^2 ou des termes similaires.
        
        # Pour que cela fonctionne avec la forme (n_features,), WX doit être un vecteur.
        # np.sum((X[:, j][:, None]**2) * (Y**2)) somme tous les éléments après la multiplication
        # Il faut sommer par dimension.
        # Une solution plus directe pour obtenir un score par caractéristique pour WX:
        # WX = np.sum((X @ Y)**2, axis=0) ou quelque chose de similaire
        # Ou comme dans certains papiers, W_j est liée à la "distance" d'une caractéristique à l'espace projeté.
        
        # En se basant sur le code original, il semble y avoir une erreur de compréhension ou une simplification drastique.
        # La ligne `WX = np.zeros(n_features)` indique que WX devrait être un vecteur.
        # Mais le calcul `np.sum((X[:, j][:, None]**2) * (Y**2))` donnera un scalaire par j si l'on boucle.
        # Pour une version plus fidèle aux papiers LLBCFS, la mise à jour de W_new est issue de la dérivée partielle.
        # Elle implique généralement le terme (X^T L X) et les embeddings Y.
        # Cependant, pour garder le code utilisable et minimiser les changements profonds,
        # je vais assumer que 'WX' doit être un score agrégé par caractéristique.
        
        # L'idée est que W doit être l'inverse de la norme des "résidus" ou d'une quantité
        # liée à la contribution de la caractéristique.
        # Une forme plus cohérente avec l'objectif de LLBCFS pour la mise à jour de W
        # serait liée à la capacité de chaque caractéristique à préserver la structure de l'embedding Y.
        
        # Si on regarde l'objectif Tr(Y.T L Y), alors la mise à jour de W devrait être:
        # Pour chaque caractéristique j, dériver la perte par rapport à W_j.
        # L'approximation donnée est W_new = 1 / (2 * sqrt(WX)).
        # WX est la somme sur les échantillons des X_ij^2 * (terme lié à Y)
        
        # Revoir la logique WX pour qu'elle produise un score par caractéristique.
        # Si (X * W) est la donnée pondérée, alors sa relation à Y est essentielle.
        # Une interprétation possible de la ligne originale:
        # WX_j = sum_i [ (X_ij * W_j)^2 * (sum_l Y_il^2) ] -- non, W_j est dedans
        # La formule W_new = 1 / (2 * np.sqrt(WX)) est standard pour le L2,1, où WX est lié aux normes des lignes.
        # WX doit être un vecteur de n_features.
        
        # Problème avec WX[j] = np.sum((X[:, j][:, None]**2) * (Y**2)) :
        # X[:, j] est (n_samples,). (X[:, j][:, None]**2) est (n_samples, 1).
        # (Y**2) est (n_samples, n_class).
        # Le produit de ces deux est (n_samples, n_class). np.sum de cela donne un scalaire.
        # Donc, la boucle avec WX[j] ferait que WX[j] serait un scalaire, et WX ne serait pas un vecteur.
        
        # J'assume que la somme doit être sur les échantillons, produisant un score par caractéristique.
        # Une interprétation: les carrés des valeurs de X_j multipliés par la "puissance" de Y
        # (représentant la pertinence de l'échantillon dans l'embedding Y)
        # Calcul des WX:
        WX_new_values = np.zeros(n_features)
        # Pour chaque caractéristique j
        for j in range(n_features):
            # Calcule la somme sur les échantillons des (X_ij)^2 * (somme_des_carrés_de_Y_il pour cet échantillon i)
            WX_new_values[j] = np.sum( (X[:, j]**2) * np.sum(Y**2, axis=1) )
        
        # Mise à jour des poids
        W_new = 1 / (2 * np.sqrt(WX_new_values + 1e-10))
        W_new /= (np.sum(W_new) + 1e-10) # Normalisation pour éviter NaN si sum est 0
        
        # 4. Vérification de la convergence et mise à jour
        obj = np.trace(Y.T @ L @ Y) + lambda1 * np.sum(np.abs(W_new)) + lambda2 * np.sum(W_new**2)
        obj_history.append(obj)
        
        if iter_count > 0 and abs(obj_history[-1] - obj_history[-2]) < tol:
            # print(f"Converged at iteration {iter_count + 1}.")
            break
        
        W = W_new
    
    # Les scores sont les poids W après convergence, triés par ordre décroissant
    return W # Retourne le vecteur de poids (scores)

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

# --- 1. Application de LLBCFS pour obtenir le classement initial ---
print("\n--- Application de l'algorithme LLBCFS pour classer toutes les caractéristiques ---")

X_llbcfs_input = df_normalized.values # LLBCFS attend (n_samples, n_features)
feature_names_llbcfs = df_normalized.columns.tolist()

# n_class est la dimension de l'espace latent pour l'embedding Y
num_clusters = len(np.unique(clusters_reference)) # Devrait être 4

# Appliquer LLBCFS
# Les hyperparamètres gamma, lambda1, lambda2, k_neighbors peuvent nécessiter un réglage fin.
scores_llbcfs = llbcfs(X_llbcfs_input, n_class=num_clusters, gamma=0.5, k_neighbors=5, lambda1=0.1, lambda2=0.01, max_iter=50)

# Création du DataFrame pour le classement complet (scores sont les poids W)
ranked_features_llbcfs = pd.Series(scores_llbcfs, index=feature_names_llbcfs).sort_values(ascending=False)

print("\nTop 10 des caractéristiques selon LLBCFS (par poids) :")
print(ranked_features_llbcfs.head(10).to_markdown(numalign="left", stralign="left"))


# --- 2. Évaluation ARI pour trouver le K optimal ---
print("\n--- Évaluation ARI pour trouver le K optimal pour LLBCFS ---")

# Définir les K valeurs à tester pour le nombre de caractéristiques
k_values_to_test = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250] # Ajustez cette liste si nécessaire

best_ari_llbcfs = -1.0
best_k_llbcfs = -1
optimal_features_list_llbcfs = [] # Stockera les noms des caractéristiques pour le K optimal

for k in k_values_to_test:
    # Sélectionner les top K caractéristiques basées sur le classement LLBCFS
    current_selected_features_names = ranked_features_llbcfs.head(k).index.tolist()

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
    if ari > best_ari_llbcfs:
        best_ari_llbcfs = ari
        best_k_llbcfs = k
        optimal_features_list_llbcfs = current_selected_features_names


# --- 3. Sauvegarde des résultats du LLBCFS ---
output_dir = "C:/Users/pc/Desktop/Features/results_fsa/"
os.makedirs(output_dir, exist_ok=True) # S'assure que le dossier existe

# Sauvegarder JUSTE les noms des caractéristiques optimales ET leurs valeurs d'importance dans un fichier CSV.
if optimal_features_list_llbcfs:
    # Sélectionner les valeurs d'importance pour les caractéristiques optimales
    final_optimal_features_series = ranked_features_llbcfs.loc[optimal_features_list_llbcfs]

    # Créer un DataFrame avec la liste des caractéristiques optimales et leurs importances
    df_optimal_features_with_values = pd.DataFrame({
        'Feature_Name': final_optimal_features_series.index,
        'LLBCFS_Importance_Value': final_optimal_features_series.values
    })
    
    optimal_features_csv_path = os.path.join(output_dir, "llbcfs_top_optimal_features.csv")
    df_optimal_features_with_values.to_csv(optimal_features_csv_path, index=False) # index=False pour ne pas écrire l'index
    print(f"\nLa liste des {len(optimal_features_list_llbcfs)} caractéristiques optimales (avec leurs valeurs d'importance LLBCFS) a été sauvegardée dans '{optimal_features_csv_path}'.")
else:
    print("\nAucune caractéristique optimale à sauvegarder dans un fichier CSV pour LLBCFS.")


print(f"\n--- Résultats Optimaux pour LLBCFS ---")
print(f"Meilleur K pour LLBCFS (selon ARI) : {best_k_llbcfs}")
print(f"ARI le plus élevé : {best_ari_llbcfs:.4f}")
print(f"Nombre de caractéristiques optimales : {len(optimal_features_list_llbcfs)}")
print(f"Noms des 10 premières caractéristiques optimales : {optimal_features_list_llbcfs[:10]}...")
print(f"Tous les résultats ont été sauvegardés dans '{output_dir}'.")

"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh

def construct_local_kernel(X, W, gamma=1.0, k=5):
    Xw = X * W  # appliquer poids à chaque feature
    K = rbf_kernel(Xw, gamma=gamma)
    G = kneighbors_graph(Xw, k, mode='connectivity', include_self=True).toarray()
    K_local = K * G
    return K_local

def laplacian_matrix(K):
    D = np.diag(np.sum(K, axis=1))
    L = D - K
    return L

def llbcfs(X, n_class, gamma=1.0, k=5, lambda1=1.0, lambda2=0.1, max_iter=20, tol=1e-5):
    n_samples, n_features = X.shape
    W = np.ones(n_features) / n_features
    
    obj_history = []
    for iter in range(max_iter):
        K_local = construct_local_kernel(X, W, gamma=gamma, k=k)
        L = laplacian_matrix(K_local)
        L = (L + L.T) / 2
        eigval, eigvec = eigh(L)
        Y = eigvec[:, :n_class]
        
        WX = np.zeros(n_features)
        for j in range(n_features):
            WX[j] = np.sum((X[:, j][:, None]**2) * (Y**2))
        W_new = 1 / (2 * np.sqrt(WX + 1e-10))
        W_new /= np.sum(W_new)
        
        obj = np.trace(Y.T @ L @ Y) + lambda1 * np.sum(np.abs(W_new)) + lambda2 * np.sum(W_new**2)
        obj_history.append(obj)
        
        if iter > 0 and abs(obj_history[-1] - obj_history[-2]) < tol:
            break
        
        W = W_new
    
    idx_sorted = np.argsort(-W)
    return W, idx_sorted, obj_history

# =============================
#  Pipeline complet LLBCFS
# =============================

# 1. Charger les données normalisées
data_path = "C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_normalized.csv"  # adapte selon ton chemin
X_df = pd.read_csv(data_path)
feature_names = X_df.columns
X_np = X_df.values

print("✅ Données normalisées chargées.")
print(f"Dimensions : {X_np.shape}")

# 2. Appliquer LLBCFS
W, selected_idx, obj_hist = llbcfs(X_np, n_class=2, gamma=0.5, k=5, lambda1=0.1, lambda2=0.01, max_iter=50)

# 3. Appliquer un seuil sur les poids (exemple 0.00001)
seuil = 0.00001
selected_features = [(feature_names[i], W[i]) for i in selected_idx if W[i] > seuil]

print(f"\nFeatures sélectionnées avec seuil > {seuil} : {len(selected_features)}")

# Affichage top 10 features sélectionnées
print("\nTop 10 features sélectionnées :")
for feat, score in selected_features[:10]:
    print(f"{feat} : {score:.6f}")

# 4. Sauvegarde
output_csv = "selected_features_by_LLBCFS.csv"
output_txt = "selected_feature_names_LLBCFS.txt"

df_out = pd.DataFrame(selected_features, columns=["Feature", "Weight"])
df_out.to_csv(output_csv, index=False)

with open(output_txt, "w") as f:
    for feat, _ in selected_features:
        f.write(f"{feat}\n")

print(f"\n📁 Résultat sauvegardé dans '{output_csv}' et '{output_txt}'")
print(f" Nombre total de caractéristiques sélectionnées : {len(selected_features)}")
"""