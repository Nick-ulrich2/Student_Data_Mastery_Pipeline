import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from options.option1.requirements.comprehension_prepa import comprehension_prepa
from options.options import get_page_options
from train import (
    aggregate_importance,
    cross_validate_candidates,
    metric_dict,
    model_candidates,
    prepare_data,
)
from themes import apply_theme, set_theme
from translations import LANGUAGES, install_ui_translation, set_language, t


st.set_page_config(
    page_title="Student_Performance_Predictor",
    page_icon="Graph/logo_spp.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)

install_ui_translation()

# Sur les petits écrans, on masque uniquement les boutons de navigation
# situés en haut : « Étape 1 » à « Étape 4 ».
st.markdown(
    """
    <style>
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"]:has(
            div[data-testid="stElementContainer"][class*="st-key-current_step_"]
        ) div[data-testid="stButton"] {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar

# Ajoutons l'image du logo sur la sidebar 
st.sidebar.image("Graph/logo_spp.webp", width=200)

previous_language = st.session_state.get("language", "fr")
language_label = st.sidebar.selectbox(
    t("Langue", "Language"),
    list(LANGUAGES),
    index=0 if st.session_state.get("language", "fr") == "fr" else 1,
    key="language_choice",
)
selected_language = LANGUAGES[language_label]
set_language(selected_language)
if selected_language != previous_language:
    st.rerun()

previous_theme = st.session_state.get("theme", "light")
theme_value = st.sidebar.radio(
    t("Thème", "Theme"),
    ["light", "dark"],
    index=0 if previous_theme == "light" else 1,
    format_func=lambda value: t("Clair", "Light") if value == "light" else t("Sombre", "Dark"),
    key="theme_choice",
)
selected_theme = "dark" if theme_value in {"dark", "Dark", "Sombre"} else "light"
set_theme(selected_theme)
apply_theme(selected_theme)

st.title("Student Performance Predictor")
st.write(
    t(
        "Cette application estime le score d’examen à partir de caractéristiques observées dans le dataset. "
        "Le résultat est une estimation statistique et ne doit pas servir seul à prendre une décision scolaire.",
        "This application estimates the exam score from observed student characteristics. "
        "The result is a statistical estimate and should not be used alone for educational decisions.",
    )
)

df, x_train, x_test, y_train, y_test = prepare_data()
TRAINING_FUNCTIONS_AVAILABLE = True

page_keys = [
    "data_preparation",
    "model_training",
    "model_evaluation",
    "final_results",
    "final_report",
]
page_labels = [
    t("Préparation et compréhension des données", "Data preparation and understanding"),
    t("Entraînement et sélection du modèle", "Model training and selection"),
    t("Évaluation et interprétation du modèle", "Model evaluation and interpretation"),
    t("Résultats, graphiques et modèle final", "Results, charts and final model"),
    t("Rapport final et traçabilité", "Final report and traceability"),
]

st.sidebar.divider()

# Navigation entre les pages
current_page_key = st.session_state.get("page_key", page_keys[0])
selected_page_label = st.sidebar.radio(
    t("Navigation", "Navigation"),
    page_labels,
    index=page_keys.index(current_page_key),
    key="page_choice",
)
page_key = page_keys[page_labels.index(selected_page_label)]
st.session_state.page_key = page_key



if page_key == "data_preparation":
    comprehension_prepa()

elif page_key == "model_training":
    st.title("Entraînement et sélection du modèle")
    st.write(
        "Cette partie explique comment plusieurs modèles sont essayés, comparés "
        "et finalement sélectionnés. L'objectif est de répondre à la question : "
        "quel modèle fonctionne le mieux ?"
    )
    st.info(
        "Un modèle n'est pas choisi parce qu'il est plus complexe, mais parce qu'il "
        "produit les erreurs les plus faibles sur les données de validation."
    )

    model_steps = [
        "6. Créer les modèles candidats",
        "7. Comparer les modèles",
        "8. Choisir le meilleur modèle",
        "9. Entraîner le modèle sélectionné",
    ]
    if "model_step" not in st.session_state:
        st.session_state.model_step = 0

    def go_to_model_step(step_index: int) -> None:
        st.session_state.model_step = step_index
        st.rerun()

    model_nav = st.columns(len(model_steps))
    for index, step_name in enumerate(model_steps):
        with model_nav[index]:
            st.button(
                f"Étape {index + 6}",
                key=f"model_nav_{index}",
                disabled=st.session_state.model_step == index,
                on_click=go_to_model_step if st.session_state.model_step != index else None,
                args=(index,) if st.session_state.model_step != index else (),
                width="stretch",
            )

    st.caption(
        f"Étape actuelle : {model_steps[st.session_state.model_step]}"
    )
    st.divider()

    if st.session_state.model_step == 0:
        st.header("6. Créer les modèles candidats")
        st.write(
            "Le programme prépare plusieurs méthodes de régression. Chaque méthode "
            "essaie de trouver une relation entre les caractéristiques des étudiants "
            "et leur score d'examen."
        )
        model_names = ["dummy_mean", "ridge", "random_forest", "extra_trees", "gradient_boosting"]
        st.dataframe(
            pd.DataFrame({
                "Modèle": model_names,
                "Rôle": [
                    "Référence basée sur la moyenne",
                    "Relation linéaire régularisée",
                    "Ensemble d'arbres aléatoires",
                    "Ensemble d'arbres très randomisés",
                    "Arbres construits progressivement",
                ],
            }),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Le modèle dummy sert de référence. Un modèle plus complexe doit faire "
            "mieux que cette référence pour être réellement utile."
        )
        if st.button("Passer à la comparaison →", type="primary", width="stretch"):
            go_to_model_step(1)

    elif st.session_state.model_step == 1:
        st.header("7. Comparer les modèles")
        st.write(
            "La validation croisée divise les données d'entraînement en plusieurs "
            "parties. Chaque modèle apprend sur certaines parties et est vérifié sur "
            "une partie différente. Cela rend la comparaison plus fiable."
        )
        if not TRAINING_FUNCTIONS_AVAILABLE:
            st.error("Les fonctions de comparaison ne sont pas disponibles dans train.py.")
        elif st.button("Lancer la validation croisée", type="primary", width="stretch"):
            candidates = model_candidates()
            comparison, cv_details = cross_validate_candidates(
                candidates,
                x_train,
                y_train,
                show_streamlit=False,
            )
            st.session_state.candidates = candidates
            st.session_state.comparison = comparison
            st.session_state.cv_details = cv_details

        comparison = st.session_state.get("comparison")
        if comparison is not None:
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.caption(
                "Le RMSE et le MAE doivent être faibles. Le R² doit généralement être "
                "le plus élevé possible."
            )
        else:
            st.info("Cliquez sur le bouton pour lancer la comparaison.")

        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button("← Revenir aux modèles", width="stretch"):
                go_to_model_step(0)
        with next_col:
            if st.button("Choisir le meilleur modèle →", type="primary", width="stretch"):
                go_to_model_step(2)

    elif st.session_state.model_step == 2:
        st.header("8. Choisir le meilleur modèle")
        st.write(
            "Les résultats sont triés selon le RMSE moyen. La première ligne contient "
            "donc le modèle qui commet, en moyenne, les erreurs les plus faibles."
        )
        comparison = st.session_state.get("comparison")
        if comparison is None:
            st.warning("Lancez d'abord la validation croisée à l'étape précédente.")
        else:
            selected_name = str(comparison.iloc[0]["model"])
            st.session_state.selected_name = selected_name
            st.success(f"Modèle retenu : {selected_name}")
            st.dataframe(comparison.head(1), width="stretch", hide_index=True)
            st.write(
                "Ce choix est basé sur une comparaison quantitative. Il ne signifie "
                "pas que les autres modèles sont toujours mauvais dans tous les cas."
            )

        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button("← Revenir à la comparaison", width="stretch"):
                go_to_model_step(1)
        with next_col:
            if st.button("Entraîner le modèle retenu →", type="primary", width="stretch"):
                go_to_model_step(3)

    else:
        st.header("9. Entraîner le modèle sélectionné")
        st.write(
            "Le modèle retenu apprend maintenant à partir des données d'entraînement. "
            "Il pourra ensuite être utilisé pour produire des prédictions sur le jeu de test."
        )
        candidates = st.session_state.get("candidates")
        selected_name = st.session_state.get("selected_name")
        if candidates is None or selected_name is None:
            st.warning("Effectuez d'abord la comparaison et la sélection du modèle.")
        elif st.button("Lancer l'entraînement", type="primary", width="stretch"):
            selected_pipeline = candidates[selected_name]
            selected_pipeline.fit(x_train, y_train)
            st.session_state.selected_pipeline = selected_pipeline
            st.success(f"Le modèle {selected_name} a été entraîné avec succès.")
            st.session_state.model_ready = True

        if st.session_state.get("model_ready", False):
            st.success("Le modèle est prêt pour l'évaluation.")
        if st.button("Recommencer le parcours", width="stretch"):
            go_to_model_step(0)

elif page_key == "model_evaluation":
    st.title("Évaluation et interprétation du modèle")
    st.write(
        "Cette partie vérifie la qualité des prédictions et cherche à comprendre "
        "quelles variables sont les plus utiles. Elle répond à la question : "
        "le modèle est-il fiable et compréhensible ?"
    )
    st.info(
        "Le modèle est évalué sur des données de test qu'il n'a pas utilisées pour apprendre."
    )

    evaluation_steps = [
        "10. Produire les prédictions",
        "11. Calculer les métriques",
        "12. Analyser les variables",
        "13. Analyser les sous-groupes",
    ]
    if "evaluation_step" not in st.session_state:
        st.session_state.evaluation_step = 0

    def go_to_evaluation_step(step_index: int) -> None:
        st.session_state.evaluation_step = step_index
        st.rerun()

    eval_nav = st.columns(len(evaluation_steps))
    for index, step_name in enumerate(evaluation_steps):
        with eval_nav[index]:
            st.button(
                f"Étape {index + 10}",
                key=f"evaluation_nav_{index}",
                disabled=st.session_state.evaluation_step == index,
                on_click=go_to_evaluation_step if st.session_state.evaluation_step != index else None,
                args=(index,) if st.session_state.evaluation_step != index else (),
                width="stretch",
            )

    st.caption(f"Étape actuelle : {evaluation_steps[st.session_state.evaluation_step]}")
    st.divider()

    model = st.session_state.get("selected_pipeline")
    if model is not None:
        test_predictions = model.predict(x_test)
        st.session_state.test_predictions = test_predictions

    if st.session_state.evaluation_step == 0:
        st.header("10. Produire les prédictions de test")
        st.write(
            "Le modèle applique ce qu'il a appris aux lignes du jeu de test. Ces lignes "
            "n'ont pas servi directement à son entraînement."
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None:
            st.warning("Entraînez d'abord un modèle dans le groupe précédent.")
        else:
            preview = pd.DataFrame({
                "Score réel": y_test.to_numpy(),
                "Score prédit": predictions,
                "Erreur absolue": np.abs(y_test.to_numpy() - predictions),
            })
            st.dataframe(preview.head(10), width="stretch", hide_index=True)
        if st.button("Calculer les métriques →", type="primary", width="stretch"):
            go_to_evaluation_step(1)

    elif st.session_state.evaluation_step == 1:
        st.header("11. Calculer les métriques")
        st.write(
            "Les métriques résument les erreurs du modèle. Le MAE donne l'erreur moyenne, "
            "le RMSE pénalise davantage les grandes erreurs et le R² mesure la variation expliquée."
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None or not TRAINING_FUNCTIONS_AVAILABLE:
            st.warning("Les prédictions et la fonction metric_dict sont nécessaires.")
        else:
            metrics = metric_dict(y_test, predictions)
            st.session_state.test_metrics = metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['mae']:.2f}")
            col2.metric("RMSE", f"{metrics['rmse']:.2f}")
            col3.metric("R²", f"{metrics['r2']:.2f}")
            st.write("Une erreur plus faible est préférable pour le MAE et le RMSE. Un R² plus élevé est généralement préférable.")
        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button("← Revenir aux prédictions", width="stretch"):
                go_to_evaluation_step(0)
        with next_col:
            if st.button("Analyser les variables →", type="primary", width="stretch"):
                go_to_evaluation_step(2)

    elif st.session_state.evaluation_step == 2:
        st.header("12. Analyser les variables importantes")
        st.write(
            "Cette analyse indique quelles variables sont le plus utilisées par le modèle. "
            "Une importance prédictive ne prouve pas qu'une variable est une cause directe du score."
        )
        model = st.session_state.get("selected_pipeline")
        if model is None or not TRAINING_FUNCTIONS_AVAILABLE:
            st.warning("Un modèle entraîné est nécessaire pour calculer les importances.")
        else:
            importances = aggregate_importance(model, show_streamlit=False)
            st.session_state.importances = importances
            if importances.empty:
                st.info("Ce modèle ne fournit pas d'importance de variables.")
            else:
                top = importances.head(12).sort_values("importance")
                st.bar_chart(top.set_index("source")["importance"])
                st.dataframe(importances.head(12), width="stretch", hide_index=True)
        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button("← Revenir aux métriques", width="stretch"):
                go_to_evaluation_step(1)
        with next_col:
            if st.button("Analyser les sous-groupes →", type="primary", width="stretch"):
                go_to_evaluation_step(3)

    else:
        st.header("13. Analyser les performances par sous-groupe")
        st.write(
            "Les performances peuvent être comparées entre différents sous-groupes. "
            "Cela permet de repérer si le modèle est beaucoup moins précis pour une catégorie particulière."
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None:
            st.warning("Produisez d'abord les prédictions de test.")
        else:
            subgroup_columns = [column for column in df.columns if df[column].dtype == "object"]
            if not subgroup_columns:
                st.info("Aucune colonne catégorielle disponible pour une analyse par sous-groupe.")
            else:
                selected_group = st.selectbox("Choisissez une colonne de groupe", subgroup_columns)
                subgroup_df = x_test.copy()
                subgroup_df["actual"] = y_test.to_numpy()
                subgroup_df["prediction"] = predictions
                subgroup_df["absolute_error"] = (subgroup_df["actual"] - subgroup_df["prediction"]).abs()
                subgroup_df[selected_group] = df.loc[x_test.index, selected_group].values
                grouped = subgroup_df.groupby(selected_group, dropna=False).agg(
                    nombre=("actual", "size"),
                    erreur_moyenne=("absolute_error", "mean"),
                    score_moyen=("actual", "mean"),
                ).reset_index()
                st.dataframe(grouped, width="stretch", hide_index=True)
        if st.button("Recommencer l'évaluation", type="primary", width="stretch"):
            go_to_evaluation_step(0)

elif page_key == "final_results":
    st.title("Résultats, graphiques et modèle final")
    st.write(
        "Cette partie rassemble les résultats produits, les graphiques d'analyse et "
        "le modèle final destiné à être réutilisé. Elle répond à la question : "
        "comment sauvegarder et réutiliser le modèle ?"
    )

    production_steps = [
        "14. Sauvegarder les résultats",
        "15. Créer les graphiques",
        "16. Réentraîner sur toutes les données",
        "17. Sauvegarder le modèle final",
    ]
    if "production_step" not in st.session_state:
        st.session_state.production_step = 0

    def go_to_production_step(step_index: int) -> None:
        st.session_state.production_step = step_index
        st.rerun()

    prod_nav = st.columns(len(production_steps))
    for index, step_name in enumerate(production_steps):
        with prod_nav[index]:
            st.button(
                f"Étape {index + 14}",
                key=f"production_nav_{index}",
                disabled=st.session_state.production_step == index,
                on_click=go_to_production_step if st.session_state.production_step != index else None,
                args=(index,) if st.session_state.production_step != index else (),
                width="stretch",
            )

    st.caption(f"Étape actuelle : {production_steps[st.session_state.production_step]}")
    st.divider()

    if st.session_state.production_step == 0:
        st.header("14. Sauvegarder les résultats")
        st.write(
            "Les résultats importants peuvent être sauvegardés dans des fichiers CSV : "
            "comparaison des modèles, importances, performances par sous-groupe et prédictions."
        )
        result_tables = {
            "Comparaison des modèles": st.session_state.get("comparison"),
            "Importance des variables": st.session_state.get("importances"),
        }
        for title, table in result_tables.items():
            if table is not None:
                st.write(f"**{title}**")
                st.dataframe(table.head(10), width="stretch", hide_index=True)
        st.info("Dans le script d'entraînement, ces tableaux sont ensuite exportés avec to_csv().")
        if st.button("Passer aux graphiques →", type="primary", width="stretch"):
            go_to_production_step(1)

    elif st.session_state.production_step == 1:
        st.header("15. Créer les graphiques")
        st.write(
            "Les graphiques permettent de comprendre la distribution de la cible, "
            "les résidus et la proximité entre scores réels et scores prédits."
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None:
            st.warning("Les prédictions de test sont nécessaires pour créer ces graphiques.")
        else:
            chart_df = pd.DataFrame({
                "Réel": y_test.to_numpy(),
                "Prédit": predictions,
            })
            st.subheader("Scores réels et prédits")
            st.scatter_chart(chart_df, x="Réel", y="Prédit")
            st.subheader("Distribution de la cible")
            st.bar_chart(df.select_dtypes(include="number").iloc[:, -1].value_counts().sort_index())
        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button("← Revenir aux résultats", width="stretch"):
                go_to_production_step(0)
        with next_col:
            if st.button("Réentraîner sur toutes les données →", type="primary", width="stretch"):
                go_to_production_step(2)

    elif st.session_state.production_step == 2:
        st.header("16. Réentraîner sur toutes les données")
        st.write(
            "Après l'évaluation, le modèle de production peut être réentraîné sur toutes "
            "les lignes disponibles. Il bénéficie ainsi du maximum d'informations avant sa sauvegarde."
        )
        if st.button("Lancer le réentraînement final", type="primary", width="stretch"):
            selected_name = st.session_state.get("selected_name")
            if selected_name and TRAINING_FUNCTIONS_AVAILABLE:
                production_model = model_candidates()[selected_name]
                x = pd.concat([x_train, x_test])
                y = pd.concat([y_train, y_test])
                production_model.fit(x, y)
                st.session_state.production_model = production_model
                st.success("Le modèle a été réentraîné sur toutes les données.")
            else:
                st.warning("Sélectionnez d'abord un modèle dans le groupe précédent.")
        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button("← Revenir aux graphiques", width="stretch"):
                go_to_production_step(1)
        with next_col:
            if st.button("Sauvegarder le modèle →", type="primary", width="stretch"):
                go_to_production_step(3)

    else:
        st.header("17. Sauvegarder le modèle final")
        st.write(
            "Le modèle final est enregistré dans un fichier afin de pouvoir être rechargé "
            "plus tard sans recommencer l'entraînement."
        )
        production_model = st.session_state.get("production_model")
        if production_model is None:
            st.warning("Réentraînez d'abord le modèle sur toutes les données.")
        elif st.button("Enregistrer le modèle final", type="primary", width="stretch"):
            model_path = Path("artifacts/models/model.joblib")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(production_model, model_path)
            st.success(f"Modèle sauvegardé dans : {model_path}")
        if st.button("Recommencer ce groupe", width="stretch"):
            go_to_production_step(0)

elif page_key == "final_report":
    st.title("Rapport final et traçabilité")
    st.write(
        "Cette dernière partie conserve les informations utiles pour comprendre, "
        "reproduire et documenter l'exécution du projet. Elle répond à la question : "
        "comment garder une trace du projet ?"
    )

    report_steps = [
        "18. Créer les métadonnées",
        "19. Enregistrer les fichiers JSON",
        "20. Afficher le résumé final",
    ]
    if "report_step" not in st.session_state:
        st.session_state.report_step = 0

    def go_to_report_step(step_index: int) -> None:
        st.session_state.report_step = step_index
        st.rerun()

    report_nav = st.columns(len(report_steps))
    for index, step_name in enumerate(report_steps):
        with report_nav[index]:
            st.button(
                f"Étape {index + 18}",
                key=f"report_nav_{index}",
                disabled=st.session_state.report_step == index,
                on_click=go_to_report_step if st.session_state.report_step != index else None,
                args=(index,) if st.session_state.report_step != index else (),
                width="stretch",
            )

    st.caption(f"Étape actuelle : {report_steps[st.session_state.report_step]}")
    st.divider()

    if st.session_state.report_step == 0:
        st.header("18. Créer les métadonnées")
        st.write(
            "Les métadonnées sont une fiche d'identité de l'entraînement. Elles peuvent "
            "indiquer le modèle utilisé, les variables, les métriques, les versions des bibliothèques "
            "et le nombre de lignes utilisées."
        )
        metadata = {
            "projet": "Student Performance Predictor",
            "variables": list(x_train.columns),
            "lignes_entraînement": int(len(x_train)),
            "lignes_test": int(len(x_test)),
            "modèle": st.session_state.get("selected_name", "non sélectionné"),
            "métriques_test": st.session_state.get("test_metrics", {}),
        }
        st.json(metadata)
        st.session_state.metadata = metadata
        if st.button("Passer à l'enregistrement JSON →", type="primary", width="stretch"):
            go_to_report_step(1)

    elif st.session_state.report_step == 1:
        st.header("19. Enregistrer les fichiers JSON")
        st.write(
            "Les informations techniques peuvent être enregistrées au format JSON. "
            "Ce format est lisible par Python et par de nombreux autres outils."
        )
        metadata = st.session_state.get("metadata", {})
        if st.button("Enregistrer les métadonnées", type="primary", width="stretch"):
            metadata_path = Path("artifacts/metadata.json")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            st.success(f"Métadonnées enregistrées dans : {metadata_path}")
        previous_col, next_col = st.columns(2)
        with previous_col:
            if st.button("← Revenir aux métadonnées", width="stretch"):
                go_to_report_step(0)
        with next_col:
            if st.button("Afficher le résumé final →", type="primary", width="stretch"):
                go_to_report_step(2)

    else:
        st.header("20. Afficher le résumé final")
        st.write(
            "Le résumé final rappelle le modèle retenu, les performances observées et "
            "les fichiers produits par le projet."
        )
        summary = {
            "modèle sélectionné": st.session_state.get("selected_name", "non disponible"),
            "métriques de test": st.session_state.get("test_metrics", {}),
            "modèle prêt à être sauvegardé": st.session_state.get("production_model") is not None,
            "rapport préparé": True,
        }
        st.json(summary)
        st.success("La traçabilité du parcours est maintenant présentée.")
        if st.button("Recommencer le rapport", type="primary", width="stretch"):
            go_to_report_step(0)

