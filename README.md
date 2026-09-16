# Student Performance Predictor

Application Streamlit de bout en bout pour explorer des données scolaires, comparer plusieurs modèles de régression et estimer le score d'examen d'un étudiant.

Le projet est conçu comme un support de portfolio et d'apprentissage : il rend visibles les étapes de préparation, d'entraînement, d'évaluation et de réutilisation d'un modèle de machine learning.

## Problème traité

Le projet cherche à estimer `Exam_Score` à partir de caractéristiques scolaires, personnelles et contextuelles observées dans un dataset d'étudiants. L'objectif est de produire une estimation reproductible et explicable, et non de remplacer une décision pédagogique ou humaine.

## Objectifs

- comprendre la structure et la qualité du dataset ;
- préparer des variables numériques et catégorielles dans une pipeline unique ;
- comparer cinq modèles de régression avec validation croisée ;
- évaluer le modèle sur un jeu de test séparé ;
- analyser les erreurs, les importances de variables et les performances par sous-groupe ;
- sauvegarder un modèle réutilisable et ses métadonnées ;
- proposer une interface Streamlit bilingue français/anglais pour la prédiction.

## Fonctionnalités

L'application contient les pages suivantes :

- **Accueil** : présentation du projet et de sa méthode ;
- **Préparation des données** : chargement, structure, qualité, analyse visuelle et séparation entraînement/test ;
- **Entraînement du modèle** : création des modèles candidats, validation croisée et sélection selon le RMSE ;
- **Évaluation du modèle** : prédictions de test, MAE, RMSE, R², résidus, importances et sous-groupes ;
- **Résultats finaux** : graphiques, réentraînement sur toutes les données et sauvegarde du modèle ;
- **Rapport et traçabilité** : métadonnées et résumé du parcours ;
- **Mode prédicteur** : formulaire de saisie, prédiction et téléchargement d'un rapport PDF.

L'application ne dépend d'aucun service externe pour afficher l'accueil ou le logo.

## Pipeline machine learning

Le flux principal est :

```text
data/SPP.csv
    -> validation et séparation des données
    -> imputation des valeurs manquantes
    -> standardisation des variables numériques
    -> one-hot encoding des variables catégorielles
    -> comparaison de cinq régressions par validation croisée
    -> évaluation sur le jeu de test
    -> réentraînement du meilleur modèle sur toutes les données
    -> artifacts/models/model.joblib
    -> prédiction depuis l'interface Streamlit
```

La préparation est encapsulée dans un `ColumnTransformer` et un `Pipeline` scikit-learn. Ainsi, le même preprocessing est utilisé à l'entraînement et à la prédiction.

## Données

Le fichier utilisé est `data/SPP.csv`.

- 6 607 lignes ;
- 21 colonnes ;
- cible : `Exam_Score` ;
- 7 variables numériques ;
- 13 variables catégorielles ;
- 157 valeurs manquantes détectées dans le fichier source ;
- aucun doublon détecté lors de l'audit ;
- plage observée de la cible dans le fichier source : 55 à 101.

Les valeurs de `Exam_Score` supérieures à 100 sont plafonnées à 100 pendant la préparation utilisée par l'entraînement et l'évaluation.

### Variables numériques

`Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Class_Size`.

### Variables catégorielles

`Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Internet_Access`, `Family_Income`, `School_Type`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`, `Electricity_Access`, `Region`, `Transport_Mode`, `Language_Section`.

## Modèles

Les modèles comparés sont :

- baseline de moyenne (`DummyRegressor`) ;
- régression Ridge ;
- Random Forest ;
- Extra Trees ;
- Gradient Boosting.

Le modèle retenu dans les métadonnées actuelles est **Ridge**. Le choix est effectué selon le RMSE moyen de la validation croisée à cinq folds.

## Résultats disponibles

Les résultats présents dans `artifacts/metadata.json` ont été calculés avec un split de test de 20 %, `random_state=42` et cinq folds de validation croisée.

Sur le jeu de test :

| Métrique | Valeur |
|---|---:|
| MAE | 0.7387 |
| RMSE | 1.9244 |
| R² | 0.7380 |

Ces mesures décrivent le run enregistré dans les artefacts. Elles ne constituent pas une garantie de performance sur une nouvelle population ou un autre dataset.

## Structure du projet

```text
.
├── app.py                         # Point d'entrée Streamlit
├── app_pages/                     # Pages de l'application
├── config.py                      # Chemins, cible et listes de features
├── train.py                       # Chargement, entraînement, évaluation et artefacts
├── pipelines.py                   # Fonctions de pipeline complémentaires
├── prepa.py                       # Préparation exploratoire historique
├── translations.py                # Traductions français/anglais
├── themes.py                      # Thèmes et couleurs des graphiques
├── data/SPP.csv                   # Dataset source
├── artifacts/                     # Modèle et métadonnées
├── Graph/                         # Logo et graphiques
├── options/                       # Composants de préparation des données
├── requirements.txt               # Dépendances Python épinglées
├── .streamlit/config.toml         # Configuration visuelle Streamlit
└── .gitignore                     # Fichiers locaux et sorties générées
```

## Technologies

- Python 3.14 ;
- Streamlit 1.61.1 ;
- pandas 3.0.5 et NumPy 2.5.2 ;
- scikit-learn 1.9.0 ;
- matplotlib et seaborn ;
- joblib pour la sérialisation du modèle ;
- ReportLab pour le rapport PDF ;
- composants Streamlit Shadcn utilisés par le prédicteur.

Les versions complètes sont épinglées dans `requirements.txt` et `libraries.txt`.

## Installation locale

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sous Windows, l'activation est généralement :

```powershell
.venv\Scripts\activate
```

L'utilisation d'un environnement virtuel est recommandée pour isoler les dépendances du projet.

## Lancer l'application

```bash
streamlit run app.py
```

L'application utilise des chemins dérivés de `config.py` et peut donc être lancée depuis un autre working directory, à condition que la racine du projet soit conservée.

## Régénérer le modèle

Le modèle versionné peut être régénéré avec :

```bash
python -c "import train; train.main()"
```

Cette commande :

1. charge `data/SPP.csv` ;
2. compare les modèles ;
3. calcule les métriques et les graphiques ;
4. entraîne le meilleur modèle sur toutes les données ;
5. écrit `artifacts/models/model.joblib` ;
6. met à jour `artifacts/models/model.sha256` ;
7. écrit les métadonnées dans `artifacts/metadata.json`.

Les tableaux et sorties intermédiaires sont générés sous `artifacts/` et sont exclus de Git lorsqu'ils sont produits automatiquement.

## Déploiement

La solution adaptée à cette application est Streamlit Community Cloud, car le projet possède un point d'entrée Streamlit unique, un modèle local sérialisé et aucune base de données ou API externe obligatoire.

Étapes :

1. pousser le projet dans un dépôt GitHub ;
2. créer une application sur Streamlit Community Cloud ;
3. sélectionner le dépôt et la branche ;
4. choisir `app.py` comme fichier principal ;
5. laisser la plateforme installer `requirements.txt` ;
6. lancer le déploiement.

Aucune variable d'environnement ni clé secrète n'est requise par le code actuel.

Après déploiement, vérifier l'accueil, la navigation, le chargement du modèle, une prédiction valide, le téléchargement PDF et le changement de langue.

## Limites connues

- le dataset est un fichier local et sa qualité conditionne directement la qualité des résultats ;
- le modèle fournit une estimation statistique, pas une décision pédagogique ;
- les importances de variables sont des associations prédictives et ne prouvent pas de causalité ;
- les métriques disponibles correspondent à un seul split de test ;
- le mode light/dark existe dans le module de thème mais le sélecteur de thème est actuellement désactivé dans l'interface principale ;
- le modèle et ses artefacts doivent être régénérés si la structure du dataset change ;
- l'interface dépend de bibliothèques de composants Streamlit pour le prédicteur PDF.

## Améliorations possibles

- ajouter des tests automatisés dédiés aux pages et aux contrats de données ;
- versionner explicitement les artefacts de résultats selon chaque entraînement ;
- ajouter une validation de schéma stricte avant toute prédiction ;
- analyser plus largement la robustesse et l'équité du modèle ;
- compléter la configuration de thème et ajouter une vérification visuelle automatisée ;
- réduire les dépendances de développement non nécessaires au runtime.

## Licence et usage

Aucune licence spécifique n'est déclarée dans le dépôt. Le projet est destiné à un usage de démonstration, d'apprentissage et de portfolio.
