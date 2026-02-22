import pandas as pd
import os
import numpy as np # Importation de numpy pour np.iscomplexobj

# --- Configuration ---
# Chemin vers le répertoire où les résultats de FSA sont sauvegardés
results_dir = "C:/Users/pc/Desktop/Features/results_fsa/"

# Liste des fichiers de résultats générés par chaque algorithme FSA
# Assurez-vous que les noms des fichiers et des colonnes de score correspondent à vos sorties
fsa_files = {
    "LASSO": "lasso_top_optimal_features.csv",
    "UMCFS": "umcfs_top_optimal_features.csv",
    "FSASL": "fsasl_top_optimal_features.csv",
    "UFSOL": "ufsol_top_optimal_features.csv",
    "UDFS": "udfs_top_optimal_features.csv",
    "LLBCFS": "llbcfs_top_optimal_features.csv",
    "ReliefF_Concept": "relieff_top_optimal_features.csv",
    "CSFA": "csfa_top_optimal_features.csv",
    "ILFS": "ilfs_top_optimal_features.csv" # Ajout de ILFS
}

# Noms des colonnes contenant les scores d'importance dans chaque fichier
# Assurez-vous que ces noms sont exacts par rapport à vos scripts précédents
score_column_names = {
    "LASSO": "Absolute_Coefficient_Value",
    "UMCFS": "UMCFS_Importance_Value",
    "FSASL": "FSASL_Importance_Value",
    "UFSOL": "UFSOL_Importance_Value",
    "UDFS": "UDFS_Importance_Value",
    "LLBCFS": "LLBCFS_Importance_Value",
    "ReliefF_Concept": "Relief_Importance_Value",
    "CSFA": "CSFA_Importance_Value",
    "ILFS": "ILFS_Importance_Value" # Ajout de la colonne de score pour ILFS
}

# --- 1. Chargement et normalisation des scores ---
print("--- Chargement et normalisation des scores des algorithmes FSA ---")

all_features_scores = {} # Dictionnaire pour stocker les scores de toutes les caractéristiques par algo

for algo_name, filename in fsa_files.items():
    file_path = os.path.join(results_dir, filename)
    if not os.path.exists(file_path):
        print(f"ATTENTION : Le fichier '{filename}' pour {algo_name} n'a pas été trouvé. Il sera ignoré.")
        continue

    df_algo = pd.read_csv(file_path)
    
    # Vérifier que la colonne de score existe
    score_col = score_column_names.get(algo_name)
    if score_col not in df_algo.columns:
        print(f"ERREUR : La colonne de score '{score_col}' est introuvable dans '{filename}' pour {algo_name}. Vérifiez le nom de la colonne.")
        continue

    # Gérer les nombres complexes pour ILFS (et potentiellement d'autres si des calculs numériques les introduisent)
    scores_series = df_algo[score_col]
    
    if pd.api.types.is_complex_dtype(scores_series):
        # Si la série est déjà de type complexe, prendre la partie réelle
        scores_series = scores_series.apply(lambda x: x.real)
    elif scores_series.dtype == 'object': 
        # Si pandas a lu les scores comme des chaînes de caractères (ex: '(18.6+0j)'),
        # tenter de les convertir en nombres complexes puis prendre la partie réelle.
        # Utiliser errors='coerce' pour gérer les valeurs non convertibles en NaN.
        try:
            scores_series = scores_series.apply(lambda x: complex(x).real if isinstance(x, str) else x)
            # Après conversion, si d'autres types sont apparus (ex: NaN), s'assurer que le dtype est numérique
            scores_series = pd.to_numeric(scores_series, errors='coerce')
        except Exception as e:
            print(f"Warning: Problème lors de la conversion des scores pour '{algo_name}' (colonne '{score_col}'). Erreur: {e}. Conversion en numérique standard.")
            scores_series = pd.to_numeric(scores_series, errors='coerce')
    else: # Si ce sont déjà des floats/ints ou d'autres types numériques simples
        scores_series = pd.to_numeric(scores_series, errors='coerce')

    # Normaliser les scores entre 0 et 1 (Min-Max Scaling) pour chaque algorithme
    min_score = scores_series.min()
    max_score = scores_series.max()
    
    # Éviter la division par zéro si tous les scores sont identiques (ex: tous à 0)
    if (max_score - min_score) == 0:
        df_algo['Normalized_Score'] = 0.0 
    else:
        df_algo['Normalized_Score'] = (scores_series - min_score) / (max_score - min_score)
    
    # Stocker les scores normalisés pour chaque caractéristique
    for index, row in df_algo.iterrows():
        feature_name = row['Feature_Name']
        normalized_score = row['Normalized_Score']
        
        # Ignorer les NaN (caractéristiques non valides ou non converties)
        if pd.isna(normalized_score):
            continue

        if feature_name not in all_features_scores:
            all_features_scores[feature_name] = {}
        all_features_scores[feature_name][algo_name] = normalized_score

print("\nScores chargés et normalisés pour chaque algorithme.")

# --- 2. Combinaison des scores ---
print("--- Combinaison des scores des caractéristiques ---")

# Créer un DataFrame à partir du dictionnaire de scores
# Remplacer les valeurs manquantes (caractéristique non sélectionnée par un algo) par 0
combined_scores_df = pd.DataFrame.from_dict(all_features_scores, orient='index').fillna(0)

# Calculer le score combiné (ici, la moyenne des scores normalisés)
# Assurez-vous que la moyenne est calculée uniquement sur les colonnes des algorithmes
algo_columns = [col for col in combined_scores_df.columns if col in fsa_files.keys()]
combined_scores_df['Combined_Score'] = combined_scores_df[algo_columns].mean(axis=1)

# Trier les caractéristiques par leur score combiné
final_ranked_features = combined_scores_df['Combined_Score'].sort_values(ascending=False)

print("\nCaractéristiques classées par score combiné.")
print("\nTop 20 des caractéristiques combinées :")
print(final_ranked_features.head(20).to_markdown(numalign="left", stralign="left"))


# --- 3. Sauvegarde des résultats combinés ---
print("\n--- Sauvegarde des résultats combinés ---")

output_filename_combined_csv = "combined_optimal_features_ranking.csv"
output_filename_combined_txt = "combined_optimal_feature_names.txt"

# Sauvegarder le classement final dans un CSV
final_ranked_features.to_csv(os.path.join(results_dir, output_filename_combined_csv), header=True, index=True, float_format='%.6f')

# Sauvegarder uniquement les noms des caractéristiques dans un fichier texte
with open(os.path.join(results_dir, output_filename_combined_txt), "w") as f:
    for feature_name in final_ranked_features.index:
        f.write(f"{feature_name}\n")

print(f"\nClassement combiné des caractéristiques sauvegardé dans '{os.path.join(results_dir, output_filename_combined_csv)}'.")
print(f"Noms des caractéristiques combinées sauvegardés dans '{os.path.join(results_dir, output_filename_combined_txt)}'.")
print(f"Nombre total de caractéristiques classées : {len(final_ranked_features)}")

# --- Prochaine étape ---
print("\n--- Prochaine étape ---")
print("Maintenant que nous avons un classement combiné des caractéristiques, la prochaine étape consiste à :")
print("1. Sélectionner un sous-ensemble des 'Top K' caractéristiques à partir de ce classement combiné.")
print("2. Utiliser ce sous-ensemble de caractéristiques pour entraîner et évaluer des modèles d'apprentissage automatique (clustering ou classification) afin de confirmer leur pertinence.")
print("3. Analyser les caractéristiques les plus importantes pour des insights cliniques si les noms des caractéristiques sont significatifs.")