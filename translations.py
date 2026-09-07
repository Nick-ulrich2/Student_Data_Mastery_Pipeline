from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

LANGUAGES = {
    "Français": "fr",
    "English": "en",
}


def get_language() -> str:
    return st.session_state.get("language", "fr")


def set_language(language: str) -> None:
    st.session_state.language = language


TRANSLATIONS = {
    "Student Performance Predictor": "Student Performance Predictor",
    "Cette application estime le score d’examen à partir de caractéristiques observées dans le dataset. Le résultat est une estimation statistique et ne doit pas servir seul à prendre une décision scolaire.": "This application estimates the exam score from observed student characteristics. The result is a statistical estimate and should not be used alone for educational decisions.",
    "Le programme prépare plusieurs méthodes de régression. Chaque méthode essaie de trouver une relation entre les caractéristiques des étudiants et leur score d'examen.": "The program prepares several regression methods. Each method tries to find a relationship between student characteristics and their exam score.",
    "La validation croisée divise les données d'entraînement en plusieurs parties. Chaque modèle apprend sur certaines parties et est vérifié sur une partie différente. Cela rend la comparaison plus fiable.": "Cross-validation splits the training data into several parts. Each model learns from some parts and is checked on a different part. This makes the comparison more reliable.",
    "Les résultats sont triés selon le RMSE moyen. La première ligne contient donc le modèle qui commet, en moyenne, les erreurs les plus faibles.": "The results are sorted by average RMSE. The first row therefore contains the model with the lowest average errors.",
    "Le modèle retenu apprend maintenant à partir des données d'entraînement. Il pourra ensuite être utilisé pour produire des prédictions sur le jeu de test.": "The selected model now learns from the training data. It can then be used to produce predictions on the test set.",
    "Cette partie vérifie la qualité des prédictions et cherche à comprendre quelles variables sont les plus utiles. Elle répond à la question : le modèle est-il fiable et compréhensible ?": "This section checks prediction quality and identifies the most useful variables. It answers the question: is the model reliable and understandable?",
    "Le modèle est évalué sur des données de test qu'il n'a pas utilisées pour apprendre.": "The model is evaluated on test data it did not use for learning.",
    "Les métriques résument les erreurs du modèle. Le MAE donne l'erreur moyenne, le RMSE pénalise davantage les grandes erreurs et le R² mesure la variation expliquée.": "Metrics summarize model errors. MAE gives the average error, RMSE penalizes larger errors more heavily, and R² measures explained variation.",
    "Cette partie rassemble les résultats produits, les graphiques d'analyse et le modèle final destiné à être réutilisé. Elle répond à la question : comment sauvegarder et réutiliser le modèle ?": "This section brings together the results, analysis charts and final model for reuse. It answers the question: how can the model be saved and reused?",
    "Les graphiques permettent de comprendre la distribution de la cible, les résidus et la proximité entre scores réels et scores prédits.": "Charts help understand the target distribution, residuals and the relationship between actual and predicted scores.",
    "Après l'évaluation, le modèle de production peut être réentraîné sur toutes les lignes disponibles. Il bénéficie ainsi du maximum d'informations avant sa sauvegarde.": "After evaluation, the production model can be retrained on all available rows. It therefore benefits from the maximum amount of information before being saved.",
    "Cette dernière partie conserve les informations utiles pour comprendre, reproduire et documenter l'exécution du projet. Elle répond à la question : comment garder une trace du projet ?": "This final section keeps the information needed to understand, reproduce and document the project run. It answers the question: how can the project be tracked?",
    "Les métadonnées sont une fiche d'identité de l'entraînement. Elles peuvent indiquer le modèle utilisé, les variables, les métriques, les versions des bibliothèques et le nombre de lignes utilisées.": "Metadata is an identity record for training. It can include the model used, variables, metrics, library versions and number of rows used.",
    "Les informations techniques peuvent être enregistrées au format JSON. Ce format est lisible par Python et par de nombreux autres outils.": "Technical information can be saved in JSON format. This format can be read by Python and many other tools.",
    "Entraînement et sélection du modèle": "Model training and selection",
    "Évaluation et interprétation du modèle": "Model evaluation and interpretation",
    "Résultats, graphiques et modèle final": "Results, charts and final model",
    "Rapport final et traçabilité": "Final report and traceability",
    "Navigation": "Navigation",
    "Langue": "Language",
    "Thème": "Theme",
    "Clair": "Light",
    "Sombre": "Dark",
    "Étape actuelle": "Current step",
    "Créer les modèles candidats": "Create candidate models",
    "Comparer les modèles": "Compare models",
    "Choisir le meilleur modèle": "Choose the best model",
    "Entraîner le modèle sélectionné": "Train the selected model",
    "Produire les prédictions de test": "Generate test predictions",
    "Calculer les métriques": "Calculate metrics",
    "Analyser les variables importantes": "Analyze important variables",
    "Analyser les performances par sous-groupe": "Analyze subgroup performance",
    "Sauvegarder les résultats": "Save results",
    "Créer les graphiques": "Create charts",
    "Réentraîner sur toutes les données": "Retrain on all data",
    "Sauvegarder le modèle final": "Save the final model",
    "Créer les métadonnées": "Create metadata",
    "Enregistrer les fichiers JSON": "Save JSON files",
    "Afficher le résumé final": "Display the final summary",
    "Passer à la comparaison →": "Go to comparison →",
    "Lancer la validation croisée": "Run cross-validation",
    "Lancer l'entraînement": "Run training",
    "Recommencer le parcours": "Restart the walkthrough",
    "Calculer les métriques →": "Calculate metrics →",
    "Analyser les variables →": "Analyze variables →",
    "Analyser les sous-groupes →": "Analyze subgroups →",
    "Recommencer l'évaluation": "Restart evaluation",
    "Passer aux graphiques →": "Go to charts →",
    "Lancer le réentraînement final": "Run final retraining",
    "Enregistrer le modèle final": "Save the final model",
    "Recommencer ce groupe": "Restart this section",
    "Passer à l'enregistrement JSON →": "Go to JSON saving →",
    "Enregistrer les métadonnées": "Save metadata",
    "Afficher le résumé final →": "Display the final summary →",
    "Recommencer le rapport": "Restart report",
    "Nombre d'étudiants": "Number of students",
    "Nombre de variables": "Number of variables",
    "Valeurs manquantes": "Missing values",
    "Lignes dupliquées": "Duplicate rows",
    "Variables numériques": "Numeric variables",
    "Variables catégorielles": "Categorical variables",
    "Modèle": "Model",
    "Rôle": "Purpose",
    "Modèle retenu": "Selected model",
    "Score réel": "Actual score",
    "Score prédit": "Predicted score",
    "Erreur absolue": "Absolute error",
    "Réel": "Actual",
    "Prédit": "Predicted",
    "Choisissez une colonne de groupe": "Choose a group column",
    "Métriques finales sur le jeu de test": "Final test-set metrics",
    "Résultats du modèle": "Model results",
    "Comparaison finale des modèles": "Final model comparison",
    "Meilleur modèle selon le RMSE :": "Best model according to RMSE:",
    "Variables les plus importantes": "Most important variables",
    "Graphiques d'analyse": "Analysis charts",
    "Distribution des notes": "Score distribution",
    "Résidus": "Residuals",
    "Scores réels et prédits": "Actual and predicted scores",
    "Importance des variables": "Feature importance",
    "feature": "Feature",
    "importance": "Importance",
    "source": "Source",
    "kind": "Type",
    "model": "Model",
    "cv_mae_mean": "CV MAE mean",
    "cv_mae_std": "CV MAE standard deviation",
    "cv_rmse_mean": "CV RMSE mean",
    "cv_rmse_std": "CV RMSE standard deviation",
    "cv_r2_mean": "CV R² mean",
    "cv_r2_std": "CV R² standard deviation",
    "group_column": "Group column",
    "group": "Group",
    "n": "Count",
    "mae": "MAE",
    "rmse": "RMSE",
    "mean_actual": "Mean actual score",
    "mean_prediction": "Mean predicted score",
    "nombre": "Count",
    "erreur_moyenne": "Mean error",
    "score_moyen": "Mean score",
    "lignes": "rows",
    "du dataset": "of the dataset",
    "Référence basée sur la moyenne": "Mean-based baseline",
    "Relation linéaire régularisée": "Regularized linear relationship",
    "Ensemble d'arbres aléatoires": "Random tree ensemble",
    "Ensemble d'arbres très randomisés": "Highly randomized tree ensemble",
    "Arbres construits progressivement": "Progressively built trees",
    "Le RMSE et le MAE doivent être faibles. Le R² doit généralement être le plus élevé possible.": "RMSE and MAE should be low. R² should generally be as high as possible.",
    "Cliquez sur le bouton pour lancer la comparaison.": "Click the button to start the comparison.",
    "← Revenir aux modèles": "← Back to models",
    "Choisir le meilleur modèle →": "Choose the best model →",
    "Lancez d'abord la validation croisée à l'étape précédente.": "Run cross-validation in the previous step first.",
    "Ce choix est basé sur une comparaison quantitative. Il ne signifie pas que les autres modèles sont toujours mauvais dans tous les cas.": "This choice is based on a quantitative comparison. It does not mean the other models are always poor in every situation.",
    "← Revenir à la comparaison": "← Back to comparison",
    "Entraîner le modèle retenu →": "Train the selected model →",
    "Effectuez d'abord la comparaison et la sélection du modèle.": "First compare and select a model.",
    "Le modèle est prêt pour l'évaluation.": "The model is ready for evaluation.",
    "Le modèle applique ce qu'il a appris aux lignes du jeu de test. Ces lignes n'ont pas servi directement à son entraînement.": "The model applies what it learned to the test set rows. These rows were not used directly for training.",
    "Entraînez d'abord un modèle dans le groupe précédent.": "Train a model in the previous section first.",
    "Les prédictions et la fonction metric_dict sont nécessaires.": "Predictions and the metric_dict function are required.",
    "Une erreur plus faible est préférable pour le MAE et le RMSE. Un R² plus élevé est généralement préférable.": "A lower error is better for MAE and RMSE. A higher R² is generally preferable.",
    "← Revenir aux prédictions": "← Back to predictions",
    "← Revenir aux métriques": "← Back to metrics",
    "Un modèle entraîné est nécessaire pour calculer les importances.": "A trained model is required to calculate feature importance.",
    "Ce modèle ne fournit pas d'importance de variables.": "This model does not provide feature importance.",
    "Produisez d'abord les prédictions de test.": "Generate test predictions first.",
    "Les performances peuvent être comparées entre différents sous-groupes. Cela permet de repérer si le modèle est beaucoup moins précis pour une catégorie particulière.": "Performance can be compared across different subgroups. This helps identify whether the model is much less accurate for a particular category.",
    "Aucune colonne catégorielle disponible pour une analyse par sous-groupe.": "No categorical column is available for subgroup analysis.",
    "Les résultats importants peuvent être sauvegardés dans des fichiers CSV : comparaison des modèles, importances, performances par sous-groupe et prédictions.": "Important results can be saved to CSV files: model comparison, feature importance, subgroup performance and predictions.",
    "Comparaison des modèles": "Model comparison",
    "Importance des variables": "Feature importance",
    "Dans le script d'entraînement, ces tableaux sont ensuite exportés avec to_csv().": "In the training script, these tables are then exported with to_csv().",
    "Les prédictions de test sont nécessaires pour créer ces graphiques.": "Test predictions are required to create these charts.",
    "Scores réels et prédits": "Actual and predicted scores",
    "Distribution de la cible": "Target distribution",
    "← Revenir aux résultats": "← Back to results",
    "Réentraîner sur toutes les données →": "Retrain on all data →",
    "Le modèle a été réentraîné sur toutes les données.": "The model was retrained on all data.",
    "Sélectionnez d'abord un modèle dans le groupe précédent.": "Select a model in the previous section first.",
    "← Revenir aux graphiques": "← Back to charts",
    "Sauvegarder le modèle →": "Save the model →",
    "Le modèle final est enregistré dans un fichier afin de pouvoir être rechargé plus tard sans recommencer l'entraînement.": "The final model is saved to a file so it can be loaded later without retraining.",
    "Réentraînez d'abord le modèle sur toutes les données.": "Retrain the model on all data first.",
    "← Revenir aux métadonnées": "← Back to metadata",
    "Le résumé final rappelle le modèle retenu, les performances observées et les fichiers produits par le projet.": "The final summary recalls the selected model, observed performance and files produced by the project.",
    "La traçabilité du parcours est maintenant présentée.": "The project traceability summary is now displayed.",
    "Cette partie explique comment plusieurs modèles sont essayés, comparés et finalement sélectionnés. L'objectif est de répondre à la question : quel modèle fonctionne le mieux ?": "This section explains how several models are tested, compared and selected. It answers the question: which model performs best?",
    "Un modèle n'est pas choisi parce qu'il est plus complexe, mais parce qu'il produit les erreurs les plus faibles sur les données de validation.": "A model is not chosen because it is more complex, but because it produces the lowest errors on validation data.",
    "Le modèle dummy sert de référence. Un modèle plus complexe doit faire mieux que cette référence pour être réellement utile.": "The dummy model is a baseline. A more complex model must outperform this baseline to be genuinely useful.",
    "Les fonctions de comparaison ne sont pas disponibles dans train.py.": "The comparison functions are not available in train.py.",
    "Le RMSE et le MAE doivent être faibles. Le R² doit généralement être le plus élevé possible.": "RMSE and MAE should be low. R² should generally be as high as possible.",
    "Les résultats sont triés selon le RMSE moyen. La première ligne contient donc le modèle qui commet, en moyenne, les erreurs les plus faibles.": "The results are sorted by average RMSE. The first row therefore contains the model with the lowest average errors.",
    "Les prédictions de test sont nécessaires pour créer ces graphiques.": "Test predictions are required to create these charts.",
    "Lancez d'abord la validation croisée à l'étape précédente.": "Run cross-validation in the previous step first.",
    "Un modèle entraîné est nécessaire pour calculer les importances.": "A trained model is required to calculate feature importance.",
    "Le modèle est prêt pour l'évaluation.": "The model is ready for evaluation.",
    "Les prédictions et la fonction metric_dict sont nécessaires.": "Predictions and the metric_dict function are required.",
    "Une erreur plus faible est préférable pour le MAE et le RMSE. Un R² plus élevé est généralement préférable.": "A lower error is better for MAE and RMSE. A higher R² is generally preferable.",
    "Cette analyse indique quelles variables sont le plus utilisées par le modèle. Une importance prédictive ne prouve pas qu'une variable est une cause directe du score.": "This analysis shows which variables are used most by the model. Predictive importance does not prove that a variable directly causes the score.",
    "Les prédictions de test sont nécessaires pour créer ces graphiques.": "Test predictions are required to create these charts.",
    "Sélectionnez d'abord un modèle dans le groupe précédent.": "Select a model in the previous section first.",
    "Réentraînez d'abord le modèle sur toutes les données.": "Retrain the model on all data first.",
    "non sélectionné": "not selected",
    "non disponible": "not available",
    "6. Créer les modèles candidats": "6. Create candidate models",
    "7. Comparer les modèles": "7. Compare models",
    "8. Choisir le meilleur modèle": "8. Choose the best model",
    "9. Entraîner le modèle sélectionné": "9. Train the selected model",
    "10. Produire les prédictions": "10. Generate predictions",
    "11. Calculer les métriques": "11. Calculate metrics",
    "12. Analyser les variables": "12. Analyze variables",
    "13. Analyser les sous-groupes": "13. Analyze subgroups",
    "14. Sauvegarder les résultats": "14. Save results",
    "15. Créer les graphiques": "15. Create charts",
    "16. Réentraîner sur toutes les données": "16. Retrain on all data",
    "17. Sauvegarder le modèle final": "17. Save the final model",
    "18. Créer les métadonnées": "18. Create metadata",
    "19. Enregistrer les fichiers JSON": "19. Save JSON files",
    "20. Afficher le résumé final": "20. Display the final summary",
    "10. Produire les prédictions de test": "10. Generate test predictions",
    "12. Analyser les variables importantes": "12. Analyze important variables",
    "13. Analyser les performances par sous-groupe": "13. Analyze subgroup performance",
    "Distribution des notes d'examens en fonction de la quantite": "Exam score distribution by quantity",
    "Matrice de correlation": "Correlation matrix",
    "projet": "Project",
    "variables": "Features",
    "lignes_entraînement": "Training rows",
    "lignes_test": "Test rows",
    "modèle": "Model",
    "métriques_test": "Test metrics",
    "modèle sélectionné": "Selected model",
    "métriques de test": "Test metrics",
    "modèle prêt à être sauvegardé": "Model ready to be saved",
    "rapport préparé": "Report prepared",
    "dummy_mean": "Mean baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "gradient_boosting": "Gradient boosting",
}


def translate_text(value: str) -> str:
    if get_language() != "en":
        return value
    translated = TRANSLATIONS.get(value, value)
    if translated != value:
        return translated
    if value.startswith("Étape "):
        if value.startswith("Étape actuelle :"):
            translated = value.replace("Étape actuelle :", "Current step:", 1)
            if " — " in translated:
                prefix, step_name = translated.split(" — ", 1)
                return f"{prefix} — {translate_text(step_name)}"
            if translated.startswith("Current step: "):
                return f"Current step: {translate_text(translated[len('Current step: '):])}"
            return translated
        return value.replace("Étape ", "Step ", 1)
    if value.startswith("Modèle retenu : "):
        return value.replace("Modèle retenu : ", "Selected model: ", 1)
    if value.startswith("Le modèle ") and value.endswith(" a été entraîné avec succès."):
        return value.replace("Le modèle ", "Model ", 1).replace(" a été entraîné avec succès.", " was trained successfully.")
    if value.startswith("Modèle sauvegardé dans : "):
        return value.replace("Modèle sauvegardé dans : ", "Model saved to: ", 1)
    if value.startswith("Métadonnées enregistrées dans : "):
        return value.replace("Métadonnées enregistrées dans : ", "Metadata saved to: ", 1)
    if value.startswith("Le modèle ") and value.endswith(" a été réentraîné sur toutes les données."):
        return value.replace("Le modèle ", "Model ", 1).replace(" a été réentraîné sur toutes les données.", " was retrained on all data.")
    if value.startswith("**") and value.endswith("**"):
        return f"**{translate_text(value[2:-2])}**"
    return value


def t(french: str, english: str) -> str:
    """Return the explicitly provided translation for shared page components."""
    return english if get_language() == "en" else french


def _translate_value(value: Any) -> Any:
    if isinstance(value, str):
        return translate_text(value)
    if isinstance(value, pd.DataFrame) and get_language() == "en":
        renamed = value.copy()
        renamed.columns = [translate_text(str(column)) for column in renamed.columns]
        for column in renamed.select_dtypes(include=["object"]).columns:
            renamed[column] = renamed[column].map(
                lambda item: translate_text(str(item)) if pd.notna(item) else item
            )
        return renamed
    if isinstance(value, (list, tuple)):
        translated = [_translate_value(item) for item in value]
        return type(value)(translated)
    if isinstance(value, dict):
        return {
            translate_text(str(key)): _translate_value(item)
            for key, item in value.items()
        }
    return value


def install_ui_translation() -> None:
    if getattr(st, "_student_translation_installed", False):
        return

    translated_methods = (
        "title", "header", "subheader", "write", "markdown", "info", "warning",
        "success", "error", "caption", "button", "selectbox", "radio", "metric",
        "dataframe", "scatter_chart", "bar_chart", "json",
    )
    for method_name in translated_methods:
        original = getattr(DeltaGenerator, method_name)

        @wraps(original)
        def translated_delta_method(*args: Any, _original=original, **kwargs: Any) -> Any:
            return _original(
                args[0],
                *tuple(_translate_value(value) for value in args[1:]),
                **{key: _translate_value(value) for key, value in kwargs.items()},
            )

        setattr(DeltaGenerator, method_name, translated_delta_method)

    for method_name in translated_methods:
        original = getattr(st, method_name)

        @wraps(original)
        def translated_root_method(*args: Any, _original=original, **kwargs: Any) -> Any:
            return _original(
                *tuple(_translate_value(value) for value in args),
                **{key: _translate_value(value) for key, value in kwargs.items()},
            )

        setattr(st, method_name, translated_root_method)
    st._student_translation_installed = True
