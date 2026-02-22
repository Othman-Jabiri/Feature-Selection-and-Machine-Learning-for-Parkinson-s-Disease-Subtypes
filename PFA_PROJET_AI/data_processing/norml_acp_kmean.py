import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os
from matplotlib import pyplot as plt
# --- 0. Pré-requis : Définir les sous-types de référence (comme dans votre script principal) ---
# Ces étapes doivent être exécutées AVANT ce script LASSO ou les résultats stockés/chargés
# Pour des raisons de démonstration, je les inclus ici, mais idéalement,
# `clusters_reference` et `df_normalized` devraient être passés ou chargés d'un fichier.

print("--- Préparation des données et définition des sous-types de référence ---")

# Chargement du fichier CSV original pour obtenir toutes les données
data_path_features = "C:/Users/pc/Desktop/PART_2_S1_AIDC/CV_TPs/PFA_PROJET_AI/data/pd_EEG_features.csv"
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
# --- 7. Visualisation (projection sur PC1 et PC2) ---
df_pca_reference["Cluster"] = clusters_reference
plt.figure(figsize=(8, 6))
plt.scatter(df_pca_reference["PC1"], df_pca_reference["PC2"], c=df_pca_reference["Cluster"], cmap='viridis')
plt.title("Clustering des patients (ACP avec 4 composantes, KMeans avec 4 clusters)")
plt.xlabel("Composante principale 1 (PC1)")
plt.ylabel("Composante principale 2 (PC2)")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()
