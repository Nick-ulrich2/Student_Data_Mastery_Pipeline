import streamlit as st
import pandas as pd
import numpy as np
from train import (
    aggregate_importance,
    cross_validate_candidates,
    metric_dict,
    model_candidates,
    prepare_data,
)
from translations import t

def model_training():
    st.title(t("Entraînement et sélection du modèle", "Model training and selection"))
    st.write(
        t(
            "Cette partie explique comment plusieurs modèles sont essayés, comparés "
            "et finalement sélectionnés. L'objectif est de répondre à la question : "
            "quel modèle fonctionne le mieux ?",
            "This section explains how several models are tried, compared "
            "and finally selected. The goal is to answer the question: "
            "which model works best?"
        )
    )
    st.info(
        t(
            "Un modèle n'est pas choisi parce qu'il est plus complexe, mais parce qu'il "
            "produit les erreurs les plus faibles sur les données de validation.",
            "A model is not chosen because it is more complex, but because it "
            "produces the lowest errors on the validation data."
        )
    )

    df, x_train, x_test, y_train, y_test = prepare_data()

    model_steps = [
        t("6. Créer les modèles candidats", "6. Create candidate models"),
        t("7. Comparer les modèles", "7. Compare models"),
        t("8. Choisir le meilleur modèle", "8. Choose the best model"),
        t("9. Entraîner le modèle sélectionné", "9. Train the selected model"),
    ]
    
    if "model_step" not in st.session_state:
        st.session_state.model_step = 0

    def go_to_model_step(step_index: int) -> None:
        st.session_state.model_step = step_index
        st.rerun()

    # Use container horizontal for better responsiveness.
    with st.container(horizontal=True):
        for index, step_name in enumerate(model_steps):
            st.button(
                f"Étape {index + 6}",
                key=f"model_nav_{index}",
                disabled=st.session_state.model_step == index,
                on_click=go_to_model_step if st.session_state.model_step != index else None,
                args=(index,) if st.session_state.model_step != index else (),
                width="stretch",
            )

    st.caption(
        f"{t('Étape actuelle', 'Current step')} : {model_steps[st.session_state.model_step]}"
    )
    st.divider()

    if st.session_state.model_step == 0:
        st.header(t("6. Créer les modèles candidats", "6. Create candidate models"))
        st.write(
            t(
                "Le programme prépare plusieurs méthodes de régression. Chaque méthode "
                "essaie de trouver une relation entre les caractéristiques des étudiants "
                "et leur score d'examen.",
                "The program prepares several regression methods. Each method "
                "tries to find a relationship between student characteristics "
                "and their exam score."
            )
        )
        model_names = ["dummy_mean", "ridge", "random_forest", "extra_trees", "gradient_boosting"]
        st.dataframe(
            pd.DataFrame({
                t("Modèle", "Model"): model_names,
                t("Rôle", "Role"): [
                    t("Référence basée sur la moyenne", "Mean-based baseline"),
                    t("Relation linéaire régularisée", "Regularized linear relationship"),
                    t("Ensemble d'arbres aléatoires", "Random forest ensemble"),
                    t("Ensemble d'arbres très randomisés", "Extremely randomized trees"),
                    t("Arbres construits progressivement", "Gradient boosted trees"),
                ],
            }),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            t(
                "Le modèle dummy sert de référence. Un modèle plus complexe doit faire "
                "mieux que cette référence pour être réellement utile.",
                "The dummy model serves as a baseline. A more complex model must perform "
                "better than this baseline to be truly useful."
            )
        )
        if st.button(t("Passer à la comparaison →", "Go to comparison →"), type="primary", width="stretch"):
            go_to_model_step(1)

    elif st.session_state.model_step == 1:
        st.header(t("7. Comparer les modèles", "7. Compare models"))
        st.write(
            t(
                "La validation croisée divise les données d'entraînement en plusieurs "
                "parties. Chaque modèle apprend sur certaines parties et est vérifié sur "
                "une partie différente. Cela rend la comparaison plus fiable.",
                "Cross-validation splits the training data into several "
                "parts. Each model learns on some parts and is verified on "
                "a different part. This makes the comparison more reliable."
            )
        )
        
        if st.button(t("Lancer la validation croisée", "Launch cross-validation"), type="primary", width="stretch"):
            with st.spinner(t("Comparaison des modèles en cours... Cela peut prendre quelques secondes.", "Comparing models... This may take a few seconds.")):
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
                t(
                    "Le RMSE et le MAE doivent être faibles. Le R² doit généralement être "
                    "le plus élevé possible.",
                    "RMSE and MAE should be low. R² should generally be "
                    "as high as possible."
                )
            )
        else:
            st.info(t("Cliquez sur le bouton pour lancer la comparaison.", "Click the button to launch comparison."))

        with st.container(horizontal=True):
            if st.button(t("← Revenir aux modèles", "← Back to models"), width="stretch"):
                go_to_model_step(0)
            if st.button(t("Choisir le meilleur modèle →", "Choose best model →"), type="primary", width="stretch"):
                go_to_model_step(2)

    elif st.session_state.model_step == 2:
        st.header(t("8. Choisir le meilleur modèle", "8. Choose the best model"))
        st.write(
            t(
                "Les résultats sont triés selon le RMSE moyen. La première ligne contient "
                "donc le modèle qui commet, en moyenne, les erreurs les plus faibles.",
                "Results are sorted by average RMSE. The first row contains "
                "the model that commits, on average, the lowest errors."
            )
        )
        comparison = st.session_state.get("comparison")
        if comparison is None:
            st.warning(t("Lancez d'abord la validation croisée à l'étape précédente.", "Launch cross-validation in the previous step first."))
        else:
            selected_name = str(comparison.iloc[0]["model"])
            st.session_state.selected_name = selected_name
            st.success(t(f"Modèle retenu : {selected_name}", f"Selected model: {selected_name}"))
            st.dataframe(comparison.head(1), width="stretch", hide_index=True)
            st.write(
                t(
                    "Ce choix est basé sur une comparaison quantitative. Il ne signifie "
                    "pas que les autres modèles sont toujours mauvais dans tous les cas.",
                    "This choice is based on a quantitative comparison. It doesn't mean "
                    "that other models are always bad in all cases."
                )
            )

        with st.container(horizontal=True):
            if st.button(t("← Revenir à la comparaison", "← Back to comparison"), width="stretch"):
                go_to_model_step(1)
            if st.button(t("Entraîner le modèle retenu →", "Train selected model →"), type="primary", width="stretch"):
                go_to_model_step(3)

    else:
        st.header(t("9. Entraîner le modèle sélectionné", "9. Train selected model"))
        st.write(
            t(
                "Le modèle retenu apprend maintenant à partir des données d'entraînement. "
                "Il pourra ensuite être utilisé pour produire des prédictions sur le jeu de test.",
                "The selected model now learns from the training data. "
                "It can then be used to produce predictions on the test set."
            )
        )
        candidates = st.session_state.get("candidates")
        selected_name = st.session_state.get("selected_name")
        if candidates is None or selected_name is None:
            st.warning(t("Effectuez d'abord la comparaison et la sélection du modèle.", "Perform comparison and model selection first."))
        elif st.button(t("Lancer l'entraînement", "Launch training"), type="primary", width="stretch"):
            with st.spinner(t("Entraînement du modèle sélectionné...", "Training the selected model...")):
                selected_pipeline = candidates[selected_name]
                selected_pipeline.fit(x_train, y_train)
                st.session_state.selected_pipeline = selected_pipeline
                st.success(t(f"Le modèle {selected_name} a été entraîné avec succès.", f"Model {selected_name} trained successfully."))
                st.session_state.model_ready = True

        if st.session_state.get("model_ready", False):
            st.success(t("Le modèle est prêt pour l'évaluation.", "Model is ready for evaluation."))
        if st.button(t("Recommencer le parcours", "Restart walkthrough"), width="stretch"):
            go_to_model_step(0)

if __name__ == "__main__":
    model_training()
