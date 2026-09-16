import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
from train import (
    model_candidates,
    prepare_data,
)
from translations import t
from config import MODEL_HASH_PATH, MODEL_PATH

def final_results():
    st.title(t("Résultats, graphiques et modèle final", "Results, charts and final model"))
    st.write(
        t(
            "Cette partie rassemble les résultats produits, les graphiques d'analyse et "
            "le modèle final destiné à être réutilisé. Elle répond à la question : "
            "comment sauvegarder et réutiliser le modèle ?",
            "This section gathers the produced results, analysis charts and "
            "the final model intended for reuse. It answers the question: "
            "how to save and reuse the model?"
        )
    )

    df, x_train, x_test, y_train, y_test = prepare_data()

    production_steps = [
        t("14. Sauvegarder les résultats", "14. Save results"),
        t("15. Graphiques d'expertise", "15. Expert charts"),
        t("16. Réentraîner sur toutes les données", "16. Retrain on all data"),
        t("17. Sauvegarder le modèle final", "17. Save final model"),
    ]
    if "production_step" not in st.session_state:
        st.session_state.production_step = 0

    def go_to_production_step(step_index: int) -> None:
        st.session_state.production_step = step_index
        st.rerun()

    with st.container(horizontal=True):
        for index, step_name in enumerate(production_steps):
            st.button(
                f"Étape {index + 14}",
                key=f"production_nav_{index}",
                disabled=st.session_state.production_step == index,
                on_click=go_to_production_step if st.session_state.production_step != index else None,
                args=(index,) if st.session_state.production_step != index else (),
                width="stretch",
            )

    st.caption(f"{t('Étape actuelle', 'Current step')} : {production_steps[st.session_state.production_step]}")
    st.divider()

    if st.session_state.production_step == 0:
        st.header(t("14. Sauvegarder les résultats", "14. Save results"))
        st.write(
            t(
                "Les résultats importants peuvent être sauvegardés dans des fichiers CSV : "
                "comparaison des modèles, importances, performances par sous-groupe et prédictions.",
                "Important results can be saved in CSV files: "
                "model comparison, importances, subgroup performance and predictions."
            )
        )
        result_tables = {
            t("Comparaison des modèles", "Model comparison"): st.session_state.get("comparison"),
            t("Importance des variables", "Feature importance"): st.session_state.get("importances"),
        }
        for title, table in result_tables.items():
            if table is not None:
                st.write(f"**{title}**")
                st.dataframe(table.head(10), width="stretch", hide_index=True)
        st.info(t("Dans le script d'entraînement, ces tableaux sont ensuite exportés avec to_csv().", "In the training script, these tables are then exported with to_csv()."))
        if st.button(t("Passer aux graphiques →", "Go to charts →"), type="primary", width="stretch"):
            go_to_production_step(1)

    elif st.session_state.production_step == 1:
        st.header(t("15. Graphiques d'expertise", "15. Expert charts"))
        st.write(
            t(
                "En tant qu'expert, je vous montre ici la réalité de la performance. "
                "Le graphique de proximité montre si le modèle sous-estime ou sur-estime les notes.",
                "As an expert, I show you the reality of performance here. "
                "The proximity chart shows if the model under- or over-estimates scores."
            )
        )
        predictions = st.session_state.get("test_predictions")
        if predictions is None:
            st.warning(t("Les prédictions de test sont nécessaires pour créer ces graphiques.", "Test predictions are required to create these charts."))
        else:
            import seaborn as sns
            import matplotlib.pyplot as plt

            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=y_test, y=predictions, scatter_kws={'alpha':0.4}, line_kws={'color':'red'}, ax=ax)
            ax.set_xlabel(t("Score Réel", "Actual Score"))
            ax.set_ylabel(t("Score Prédit", "Predicted Score"))
            ax.set_title(t("Régression : Réel vs Prédit", "Regression: Actual vs Predicted"))
            st.pyplot(fig)

            # Error distribution
            st.subheader(t("Distribution de l'Erreur", "Error Distribution"))
            errors = y_test.to_numpy() - predictions
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.histplot(errors, kde=True, ax=ax2, color="purple")
            ax2.set_title(t("Distribution des résidus", "Residuals Distribution"))
            st.pyplot(fig2)
        with st.container(horizontal=True):
            if st.button(t("← Revenir aux résultats", "← Back to results"), width="stretch"):
                go_to_production_step(0)
            if st.button(t("Réentraîner sur toutes les données →", "Retrain on all data →"), type="primary", width="stretch"):
                go_to_production_step(2)

    elif st.session_state.production_step == 2:
        st.header(t("16. Réentraîner sur toutes les données", "16. Retrain on all data"))
        st.write(
            t(
                "Après l'évaluation, le modèle de production peut être réentraîné sur toutes "
                "les lignes disponibles. Il bénéficie ainsi du maximum d'informations avant sa sauvegarde.",
                "After evaluation, the production model can be retrained on all "
                "available rows. It thus benefits from maximum information before being saved."
            )
        )
        if st.button(t("Lancer le réentraînement final", "Launch final retraining"), type="primary", width="stretch"):
            with st.spinner(t("Réentraînement sur l'intégralité des données...", "Retraining on all data...")):
                selected_name = st.session_state.get("selected_name")
                if selected_name:
                    production_model = model_candidates()[selected_name]
                    x = pd.concat([x_train, x_test])
                    y = pd.concat([y_train, y_test])
                    production_model.fit(x, y)
                    st.session_state.production_model = production_model
                    st.success(t("Le modèle a été réentraîné sur toutes les données.", "The model has been retrained on all data."))
                else:
                    st.warning(t("Sélectionnez d'abord un modèle dans le groupe précédent.", "Select a model in the previous group first."))
        with st.container(horizontal=True):
            if st.button(t("← Revenir aux graphiques", "← Back to charts"), width="stretch"):
                go_to_production_step(1)
            if st.button(t("Sauvegarder le modèle →", "Save model →"), type="primary", width="stretch"):
                go_to_production_step(3)

    else:
        st.header(t("17. Sauvegarder le modèle final", "17. Save final model"))
        st.write(
            t(
                "Le modèle final est enregistré dans un fichier afin de pouvoir être rechargé "
                "plus tard sans recommencer l'entraînement.",
                "The final model is saved to a file so it can be reloaded "
                "later without starting training again."
            )
        )
        production_model = st.session_state.get("production_model")
        if production_model is None:
            st.warning(t("Réentraînez d'abord le modèle sur toutes les données.", "Retrain the model on all data first."))
        elif st.button(t("Enregistrer le modèle final", "Save final model"), type="primary", width="stretch"):
            try:
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(production_model, MODEL_PATH)
                
                # Calcul du hash pour la sécurité
                import hashlib
                sha256_hash = hashlib.sha256()
                with open(MODEL_PATH, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                
                MODEL_HASH_PATH.write_text(sha256_hash.hexdigest())
                
                st.success(f"{t('Modèle sauvegardé et sécurisé dans', 'Model saved and secured in')} : {MODEL_PATH}")
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde : {e}")
        if st.button(t("Recommencer ce groupe", "Restart this group"), width="stretch"):
            go_to_production_step(0)

if __name__ == "__main__":
    final_results()
