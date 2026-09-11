import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from train import (
    aggregate_importance,
    metric_dict,
    prepare_data,
)
from translations import t

def model_evaluation():
    st.title(t("Évaluation et interprétation du modèle", "Model evaluation and interpretation"))
    st.write(
        t(
            "Cette partie vérifie la qualité des prédictions et cherche à comprendre "
            "quelles variables sont les plus utiles. Elle répond à la question : "
            "le modèle est-il fiable et compréhensible ?",
            "This section checks prediction quality and seeks to understand "
            "which variables are most useful. It answers the question: "
            "is the model reliable and understandable?"
        )
    )
    st.info(
        t(
            "Le modèle est évalué sur des données de test qu'il n'a pas utilisées pour apprendre.",
            "The model is evaluated on test data that it didn't use to learn."
        )
    )

    df, x_train, x_test, y_train, y_test = prepare_data()

    evaluation_steps = [
        t("10. Produire les prédictions", "10. Produce predictions"),
        t("11. Calculer les métriques", "11. Calculate metrics"),
        t("12. Analyse des erreurs (Résidus)", "12. Error analysis (Residuals)"),
        t("13. Analyser les variables", "13. Analyze variables"),
        t("14. Analyser les sous-groupes", "14. Analyze subgroups"),
    ]
    if "evaluation_step" not in st.session_state:
        st.session_state.evaluation_step = 0

    def go_to_evaluation_step(step_index: int) -> None:
        st.session_state.evaluation_step = step_index
        st.rerun()

    with st.container(horizontal=True):
        for index, step_name in enumerate(evaluation_steps):
            st.button(
                f"Étape {index + 10}",
                key=f"evaluation_nav_{index}",
                disabled=st.session_state.evaluation_step == index,
                on_click=go_to_evaluation_step if st.session_state.evaluation_step != index else None,
                args=(index,) if st.session_state.evaluation_step != index else (),
                width="stretch",
            )

    st.caption(f"{t('Étape actuelle', 'Current step')} : {evaluation_steps[st.session_state.evaluation_step]}")
    st.divider()

    model = st.session_state.get("selected_pipeline")
    if model is not None:
        test_predictions = model.predict(x_test)
        st.session_state.test_predictions = test_predictions

    if st.session_state.evaluation_step == 0:
        st.header(t("10. Produire les prédictions de test", "10. Produce test predictions"))
        st.write(
            t(
                "Le modèle applique ce qu'il a appris aux lignes du jeu de test. Ces lignes "
                "n'ont pas servi directement à son entraînement.",
                "The model applies what it learned to the test set rows. These rows "
                "were not used directly for its training."
            )
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None:
            st.warning(t("Entraînez d'abord un modèle dans le groupe précédent.", "Train a model in the previous group first."))
        else:
            preview = pd.DataFrame({
                t("Score réel", "Actual score"): y_test.to_numpy(),
                t("Score prédit", "Predicted score"): predictions,
                t("Erreur absolue", "Absolute error"): np.abs(y_test.to_numpy() - predictions),
            })
            st.dataframe(preview.head(10), width="stretch", hide_index=True)
        if st.button(t("Calculer les métriques →", "Calculate metrics →"), type="primary", width="stretch"):
            go_to_evaluation_step(1)

    elif st.session_state.evaluation_step == 1:
        st.header(t("11. Calculer les métriques", "11. Calculate metrics"))
        st.write(
            t(
                "Les métriques résument les erreurs du modèle. Le MAE donne l'erreur moyenne, "
                "le RMSE pénalise davantage les grandes erreurs et le R² mesure la variation expliquée.",
                "Metrics summarize model errors. MAE gives the mean error, "
                "RMSE penalizes large errors more, and R² measures the explained variation."
            )
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None:
            st.warning(t("Les prédictions sont nécessaires.", "Predictions are required."))
        else:
            metrics = metric_dict(y_test, predictions)
            st.session_state.test_metrics = metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['mae']:.2f}")
            col2.metric("RMSE", f"{metrics['rmse']:.2f}")
            col3.metric("R²", f"{metrics['r2']:.2f}")
            st.write(t("Une erreur plus faible est préférable pour le MAE et le RMSE. Un R² plus élevé est généralement préférable.", "A lower error is preferable for MAE and RMSE. A higher R² is generally preferable."))
        with st.container(horizontal=True):
            if st.button(t("← Revenir aux prédictions", "← Back to predictions"), width="stretch"):
                go_to_evaluation_step(0)
            if st.button(t("Analyse des erreurs →", "Error analysis →"), type="primary", width="stretch"):
                go_to_evaluation_step(2)

    elif st.session_state.evaluation_step == 2:
        st.header(t("12. Analyse des erreurs (Résidus)", "12. Error Analysis (Residuals)"))
        st.write(
            t(
                "L'analyse des résidus (différence entre réel et prédit) permet de voir si le modèle "
                "a un biais systématique. Idéalement, les points devraient être répartis aléatoirement autour de 0.",
                "Residual analysis (difference between actual and predicted) helps see if the model "
                "has a systematic bias. Ideally, points should be randomly distributed around 0."
            )
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is not None:
            residuals = y_test.to_numpy() - predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=predictions, y=residuals, alpha=0.5, ax=ax)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel(t("Prédictions", "Predictions"))
            ax.set_ylabel(t("Résidus", "Residuals"))
            st.pyplot(fig)
            
            st.info(t("Si vous voyez une forme (courbe, entonnoir), cela signifie que le modèle oublie une information structurante.", 
                      "If you see a shape (curve, funnel), it means the model is missing structuring information."))
            
        with st.container(horizontal=True):
            if st.button(t("← Revenir aux métriques", "← Back to metrics"), width="stretch"):
                go_to_evaluation_step(1)
            if st.button(t("Analyser les variables →", "Analyze variables →"), type="primary", width="stretch"):
                go_to_evaluation_step(3)

    elif st.session_state.evaluation_step == 3:
        st.header(t("13. Analyser les variables importantes", "13. Analyze important variables"))
        st.write(
            t(
                "Cette analyse indique quelles variables sont le plus utilisées par le modèle. "
                "Une importance prédictive ne prouve pas qu'une variable est une cause directe du score.",
                "This analysis shows which variables are used most by the model. "
                "Predictive importance does not prove that a variable is a direct cause of the score."
            )
        )
        model = st.session_state.get("selected_pipeline")
        if model is None:
            st.warning(t("Un modèle entraîné est nécessaire pour calculer les importances.", "A trained model is required to compute importances."))
        else:
            importances = aggregate_importance(model, show_streamlit=False)
            st.session_state.importances = importances
            if importances.empty:
                st.info(t("Ce modèle ne fournit pas d'importance de variables.", "This model does not provide feature importances."))
            else:
                top = importances.head(12).sort_values("importance")
                st.bar_chart(top.set_index("source")["importance"])
                st.dataframe(importances.head(12), width="stretch", hide_index=True)
        with st.container(horizontal=True):
            if st.button(t("← Revenir aux erreurs", "← Back to errors"), width="stretch"):
                go_to_evaluation_step(2)
            if st.button(t("Analyser les sous-groupes →", "Analyze subgroups →"), type="primary", width="stretch"):
                go_to_evaluation_step(4)

    else:
        st.header(t("14. Analyser les performances par sous-groupe", "14. Analyze subgroup performances"))
        st.write(
            t(
                "Les performances peuvent être comparées entre différents sous-groupes. "
                "Cela permet de repérer si le modèle est beaucoup moins précis pour une catégorie particulière.",
                "Performances can be compared between different subgroups. "
                "This helps identify if the model is significantly less accurate for a specific category."
            )
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None:
            st.warning(t("Produisez d'abord les prédictions de test.", "Produce test predictions first."))
        else:
            subgroup_columns = [column for column in df.columns if df[column].dtype == "object"]
            if not subgroup_columns:
                st.info(t("Aucune colonne catégorielle disponible pour une analyse par sous-groupe.", "No categorical columns available for subgroup analysis."))
            else:
                selected_group = st.selectbox(t("Choisissez une colonne de groupe", "Choose a group column"), subgroup_columns)
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
        if st.button(t("Recommencer l'évaluation", "Restart evaluation"), type="primary", width="stretch"):
            go_to_evaluation_step(0)

if __name__ == "__main__":
    model_evaluation()
