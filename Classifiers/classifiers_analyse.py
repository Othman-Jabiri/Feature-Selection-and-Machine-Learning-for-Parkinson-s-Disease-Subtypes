import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings

# --- Importation des 9 classifieurs ---
from sklearn.tree import DecisionTreeClassifier # (1) DTC
from sklearn.svm import SVC # (2) Lib_SVM
from sklearn.neighbors import KNeighborsClassifier # (3) KNNC

# (4) ELC (Classificateur d'ensemble plus simple) - Implémentation générique d'ensemble
# Pour cet exemple, nous utiliserons un BaggingClassifier avec un arbre de décision comme base.
from sklearn.ensemble import BaggingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # (5) LDAC

# (6) Nouveau PNNC (Nouveau classificateur de réseau neuronal probabiliste)
# Scikit-learn n'a pas de PNN direct. MLPClassifier avec probabilités est une alternative fonctionnelle,
# mais ce n'est pas un PNN au sens strict. Pour un PNN pur, une implémentation custom ou une autre lib serait nécessaire.
from sklearn.neural_network import MLPClassifier

# (7) ECOCMC (Classificateur de modèles de codes de sortie à correction d'erreurs)
# Scikit-learn implémente cela via OutputCodeClassifier. Nécessite un estimateur de base.
from sklearn.multiclass import OutputCodeClassifier

# (8) MLP_BPC (Classificateur de rétropropagation perceptron multicouche)
# C'est simplement MLPClassifier avec une configuration spécifique de couches. Déjà importé.

from sklearn.ensemble import RandomForestClassifier # (9) RFC

# Ignorer les avertissements pour une sortie plus propre, surtout avec MLP ou SVC qui peuvent en générer
warnings.filterwarnings('ignore')

print("Bibliothèques importées avec succès.")



print("\n--- 2.1. Chargement des Données et des Caractéristiques Optimales ---")

# --- A. Chargement de votre dataset ---
# Remplacez 'your_dataset_with_Identified_Subtype.csv' par le nom exact de votre fichier CSV.
# Assurez-vous que ce fichier contient toutes les caractéristiques originales ET la colonne 'Identified_Subtype'.
data_file_path = 'C:/Users/pc/Desktop/Features/results_fsa/patients_with_identified_subtypes.csv'
try:
    df_full = pd.read_csv(data_file_path)
    print(f"Dataset '{data_file_path}' chargé avec succès. Dimensions : {df_full.shape}")
except FileNotFoundError:
    print(f"Erreur : Le fichier '{data_file_path}' n'a pas été trouvé.")
    print("Création d'un dataset factice pour démonstration. Veuillez remplacer par votre vrai fichier.")
    # --- Création d'un dataset factice pour que le code puisse s'exécuter ---
    n_samples = 300
    n_total_features = 752 # Nombre total de caractéristiques
    np.random.seed(42)
    # Simuler des caractéristiques numériques
    data_values = np.random.rand(n_samples, n_total_features) * 100
    df_full = pd.DataFrame(data_values, columns=[f'feature_{i:03d}' for i in range(n_total_features)])
    # Simuler la colonne Identified_Subtype (par exemple, 4 sous-types)
    df_full['Identified_Subtype'] = np.random.randint(0, 4, n_samples)
    # Ajouter quelques colonnes non-features pour l'exemple
    df_full['patient_id'] = range(n_samples)
    df_full['gender'] = np.random.choice(['M', 'F'], n_samples)
    df_full['age'] = np.random.randint(40, 80, n_samples)

print("\nAperçu des premières lignes du dataset :")
print(df_full.head().to_markdown(index=False, numalign="left", stralign="left"))
print("\nInformations sur le dataset (colonnes et types) :")
print(df_full.info())

# --- B. Chargement des noms des caractéristiques optimales ---
# Le fichier 'final_optimal_features_names.txt' doit contenir un nom de caractéristique par ligne.
optimal_features_file = 'C:/Users/pc/Desktop/Features/results_fsa/final_optimal_features_names.txt'
try:
    with open(optimal_features_file, 'r') as f:
        optimal_features = [line.strip() for line in f if line.strip()]
    print(f"\n{len(optimal_features)} caractéristiques optimales chargées depuis '{optimal_features_file}'.")
    # Vérifier que les caractéristiques chargées sont bien dans le DataFrame
    missing_features = [f for f in optimal_features if f not in df_full.columns]
    if missing_features:
        print(f"ATTENTION : Les caractéristiques suivantes sont manquantes dans le dataset : {missing_features}")
        # Filtrer les caractéristiques manquantes si nécessaire pour éviter des erreurs
        optimal_features = [f for f in optimal_features if f in df_full.columns]
        print(f"Utilisation de {len(optimal_features)} caractéristiques disponibles.")
except FileNotFoundError:
    print(f"Erreur : Le fichier '{optimal_features_file}' n'a pas été trouvé.")
    print("Génération d'une liste factice de 25 caractéristiques pour démonstration.")
    # --- Génération d'une liste factice de caractéristiques optimales si le fichier n'est pas trouvé ---
    # Cette partie simule le résultat de votre Étape 1
    all_possible_features = [col for col in df_full.columns if col not in ['Identified_Subtype', 'patient_id', 'gender', 'age']]
    if len(all_possible_features) >= 25:
        np.random.seed(42) # Pour la reproductibilité de la sélection factice
        optimal_features = np.random.choice(all_possible_features, 25, replace=False).tolist()
    else:
        optimal_features = all_possible_features # Si moins de 25, prendre toutes les features disponibles
    print(f"Liste factice de {len(optimal_features)} caractéristiques optimales créée.")

print("\nExemple des 5 premières caractéristiques optimales :")
print(optimal_features[:5])





# --- C. Définition de X (caractéristiques) et y (cible) ---
target_column = 'Identified_Subtype'

X = df_full[optimal_features]
y = df_full[target_column]

print(f"\nDimensions finales de X (caractéristiques sélectionnées) : {X.shape}")
print(f"Dimensions de y (cible '{target_column}') : {y.shape}")

# Vérification des valeurs uniques dans la cible
print(f"Sous-types uniques identifiés : {y.unique()}")
print(f"Distribution des sous-types :\n{y.value_counts()}")








print("\n--- 2.2. Division des Données et Normalisation (pour la Cross-Validation) ---")

# Initialisation du scaler (sera entraîné à chaque pli de la CV)
# On le définit ici pour montrer qu'il est réinitialisé à chaque pli si l'on voulait,
# mais pour une simple split train/test, on le ferait une seule fois.
# Pour la CV, le scaler.fit_transform() est fait sur le train set de chaque pli.

# Définition de la stratégie de validation croisée stratifiée
# Assure que la proportion des classes est la même dans chaque pli.
n_splits = 5 # Typiquement 5 ou 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Stratégie de validation croisée : {n_splits}-Fold Stratified K-Fold.")






print("\n--- 2.3. Définition des 9 Classifieurs ---")

classifiers = {
    "DTC": DecisionTreeClassifier(random_state=42),
    "Lib_SVM": SVC(probability=True, random_state=42), # probability=True est utile pour ROC AUC
    "KNNC": KNeighborsClassifier(),
    "ELC": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=100, random_state=42),
    # ELC: Simple Bagging avec un DTC comme estimateur de base, 100 arbres.
    "LDAC": LinearDiscriminantAnalysis(),
    "Nouveau PNNC": MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam',
                                  max_iter=300, random_state=42, learning_rate_init=0.01,
                                  # On utilise un MLP pour simuler un NN probabiliste car pas de PNN direct dans sklearn
                                  # max_iter augmenté pour la convergence
                                  ),
    # ECOCMC: Exige un estimateur de base. Ici, un SVC avec kernel linéaire.
    "ECOCMC": OutputCodeClassifier(estimator=SVC(kernel='linear', random_state=42), code_size=2, random_state=42),
    "MLP_BPC": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                             max_iter=500, random_state=42), # Plus de couches/neurones, plus d'itérations
    "RFC": RandomForestClassifier(n_estimators=100, random_state=42)
}

print(f"Liste des classifieurs définis : {list(classifiers.keys())}")







from sklearn.model_selection import cross_validate

print("\n--- 2.4. Entraînement et Évaluation des Classifieurs avec Validation Croisée ---")

# Métriques à collecter
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted'
}

all_results = {}

for name, classifier in classifiers.items():
    print(f"\nÉvaluation de : {name}")

    # Pour chaque pli de la validation croisée, nous effectuons :
    # 1. Normalisation (fit sur train, transform sur train et test)
    # 2. Entraînement du modèle
    # 3. Prédiction et calcul des métriques

    # cross_validate gère la division, la normalisation à l'intérieur de chaque pli, et le calcul des métriques
    # Il est crucial de s'assurer que le scaler est fit_transform sur le train_fold et transform sur le test_fold
    # La manière standard de faire cela est d'utiliser un Pipeline.

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Le scaler est fit à chaque pli sur le jeu d'entraînement du pli
        ('classifier', classifier)
    ])

    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=-1) # n_jobs=-1 utilise tous les cœurs

    # Stockage des moyennes des métriques
    all_results[name] = {
        'Accuracy (moyenne)': np.mean(cv_results['test_accuracy']),
        'Precision (moyenne pondérée)': np.mean(cv_results['test_precision_weighted']),
        'Recall (moyenne pondérée)': np.mean(cv_results['test_recall_weighted']),
        'F1-Score (moyenne pondérée)': np.mean(cv_results['test_f1_weighted']),
        'Temps d\'entraînement (s)': np.mean(cv_results['fit_time']),
        'Temps de prédiction (s)': np.mean(cv_results['score_time'])
    }

    print(f"  Accuracy moyenne: {all_results[name]['Accuracy (moyenne)']:.4f}")
    print(f"  F1-Score moyen: {all_results[name]['F1-Score (moyenne pondérée)']:.4f}")

# --- Affichage des Résultats Consolidés ---
print("\n--- Résultats Comparatifs Détaillés (Moyennes sur Validation Croisée) ---")
performance_df = pd.DataFrame.from_dict(all_results, orient='index')
# Formatter pour une meilleure lisibilité
pd.options.display.float_format = '{:,.4f}'.format
print(performance_df.to_markdown(numalign="left", stralign="left"))

# Optionnel : Sauvegarder les résultats dans un fichier CSV
# results_output_path = 'classification_results_summary.csv'
# performance_df.to_csv(results_output_path)
# print(f"\nRésumé des performances sauvegardé dans '{results_output_path}'")

print("\n--- Étape 2 : Prédiction des Sous-types de MP terminée ---")










 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Assurez-vous que 'performance_df' et 'results' sont définis ---
# Si vous exécutez ce bloc de code séparément, vous devrez re-exécuter les parties précédentes
# qui définissent 'performance_df' et 'results'. Pour une exécution autonome à des fins de test:
# --- DUMMY DATA FOR DEMONSTRATION IF RUNNING THIS BLOCK ALONE ---
if 'performance_df' not in locals():
    print("WARNING: 'performance_df' not found. Creating dummy data for demonstration.")
    data = {
        'DTC': {'Accuracy (moyenne)': 0.907442, 'Precision (moyenne pondérée)': 0.909997, 'Recall (moyenne pondérée)': 0.907442, 'F1-Score (moyenne pondérée)': 0.907441, 'Temps d\'entraînement (s)': 0.028023, 'Temps de prédiction (s)': 0.0187613},
        'Lib_SVM': {'Accuracy (moyenne)': 0.961685, 'Precision (moyenne pondérée)': 0.962495, 'Recall (moyenne pondérée)': 0.961685, 'F1-Score (moyenne pondérée)': 0.961299, 'Temps d\'entraînement (s)': 0.10417, 'Temps de prédiction (s)': 0.0239947},
        'KNNC': {'Accuracy (moyenne)': 0.94844, 'Precision (moyenne pondérée)': 0.949173, 'Recall (moyenne pondérée)': 0.94844, 'F1-Score (moyenne pondérée)': 0.948413, 'Temps d\'entraînement (s)': 0.0167719, 'Temps de prédiction (s)': 0.441901},
        'ELC': {'Accuracy (moyenne)': 0.948449, 'Precision (moyenne pondérée)': 0.949914, 'Recall (moyenne pondérée)': 0.948449, 'F1-Score (moyenne pondérée)': 0.94829, 'Temps d\'entraînement (s)': 1.88314, 'Temps de prédiction (s)': 0.0768506},
        'LDAC': {'Accuracy (moyenne)': 0.918029, 'Precision (moyenne pondérée)': 0.921807, 'Recall (moyenne pondérée)': 0.918029, 'F1-Score (moyenne pondérée)': 0.918156, 'Temps d\'entraînement (s)': 0.0342136, 'Temps de prédiction (s)': 0.0213006},
        'Nouveau PNNC': {'Accuracy (moyenne)': 0.957659, 'Precision (moyenne pondérée)': 0.959715, 'Recall (moyenne pondérée)': 0.957659, 'F1-Score (moyenne pondérée)': 0.957382, 'Temps d\'entraînement (s)': 0.381418, 'Temps de prédiction (s)': 0.0278417},
        'ECOCMC': {'Accuracy (moyenne)': 0.966957, 'Precision (moyenne pondérée)': 0.9673, 'Recall (moyenne pondérée)': 0.966957, 'F1-Score (moyenne pondérée)': 0.966861, 'Temps d\'entraînement (s)': 0.0739169, 'Temps de prédiction (s)': 0.0265219},
        'MLP_BPC': {'Accuracy (moyenne)': 0.96565, 'Precision (moyenne pondérée)': 0.967974, 'Recall (moyenne pondérée)': 0.96565, 'F1-Score (moyenne pondérée)': 0.965471, 'Temps d\'entraînement (s)': 1.22313, 'Temps de prédiction (s)': 0.0185669},
        'RFC': {'Accuracy (moyenne)': 0.951107, 'Precision (moyenne pondérée)': 0.952601, 'Recall (moyenne pondérée)': 0.951107, 'F1-Score (moyenne pondérée)': 0.95108, 'Temps d\'entraînement (s)': 0.608908, 'Temps de prédiction (s)': 0.03203}
    }
    performance_df = pd.DataFrame.from_dict(data, orient='index')

if 'results' not in locals():
    print("WARNING: 'results' not found. Creating dummy data for confusion matrices.")
    # Dummy data for confusion matrices (assuming 4 classes)
    results = {
        "ECOCMC": {"Confusion Matrix": np.array([[50, 1, 0, 0], [0, 48, 2, 0], [1, 0, 49, 0], [0, 0, 0, 50]])},
        "MLP_BPC": {"Confusion Matrix": np.array([[49, 2, 0, 0], [0, 47, 3, 0], [2, 0, 48, 0], [0, 1, 0, 49]])},
        "Lib_SVM": {"Confusion Matrix": np.array([[48, 3, 0, 0], [1, 49, 0, 0], [0, 1, 49, 0], [0, 0, 1, 49]])}
    }
if 'y' not in locals(): # Dummy y for class_labels
    print("WARNING: 'y' not found. Creating dummy 'y' for class labels.")
    y = pd.Series(np.random.randint(0, 4, 100)) # 4 classes, 100 samples


print("\n--- Génération des Graphiques de Performance ---")


# --- 1. Diagramme à barres comparatif des métriques clés ---

# Sélection des métriques à visualiser
metrics_to_plot = ['Accuracy (moyenne)', 'F1-Score (moyenne pondérée)',
                   'Precision (moyenne pondérée)', 'Recall (moyenne pondérée)']

# Création du graphique
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12)) # 2x2 subplots
axes = axes.flatten() # Pour itérer facilement sur les axes

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    # Tri par la métrique actuelle pour une meilleure visualisation
    sorted_df = performance_df.sort_values(by=metric, ascending=False)
    sns.barplot(x=sorted_df.index, y=sorted_df[metric], ax=ax, palette='viridis')
    ax.set_title(f'{metric} par Classifieur', fontsize=14)
    ax.set_xlabel('Classifieur', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0.85, 1.0) # Ajuster la limite y pour mieux voir les différences
    
    # --- CORRECTION ICI : Utiliser set_xticklabels pour la rotation et l'alignement ---
    ax.set_xticklabels(sorted_df.index, rotation=45, ha='right', fontsize=10) # Rotation et alignement

    for p in ax.patches: # Afficher les valeurs sur les barres
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.suptitle("Comparaison des Performances des Classifieurs", y=1.02, fontsize=16) # Titre général
plt.savefig('comparaison_performance_classifieurs.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGraphique de comparaison des performances enregistré sous 'comparaison_performance_classifieurs.png'")

# --- 2. Heatmap des Matrices de Confusion pour les meilleurs classifieurs ---

# Identifiez les N meilleurs classifieurs pour la visualisation des matrices de confusion
# Par exemple, les 3 meilleurs classifieurs basés sur le F1-Score
top_n_classifiers_names = performance_df.sort_values(by='F1-Score (moyenne pondérée)', ascending=False).head(3).index.tolist()

print(f"\nGénération des Heatmaps des Matrices de Confusion pour les meilleurs classifieurs : {top_n_classifiers_names}")

# Assurez-vous que les labels de classe sont disponibles (les sous-types uniques)
# Ils viennent de y.unique()
class_labels = np.sort(y.unique()).astype(str) # Convertir en string si ce sont des int/float

for name in top_n_classifiers_names:
    # Récupérer la matrice de confusion du dictionnaire 'results'
    # La matrice de confusion est stockée dans 'results[name]["Confusion Matrix"]'
    cm = results[name]["Confusion Matrix"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Matrice de Confusion pour {name}', fontsize=16)
    plt.xlabel('Classes Prédites', fontsize=12)
    plt.ylabel('Classes Réelles', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300)
    plt.show()

print("\nHeatmaps des matrices de confusion enregistrées.")