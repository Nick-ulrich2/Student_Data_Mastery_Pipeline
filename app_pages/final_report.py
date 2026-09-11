import streamlit as st
import json
from pathlib import Path
from train import prepare_data
from translations import t

def final_report():
    st.title(t("Rapport final et traçabilité", "Final report and traceability"))
    st.write(
        t(
            "Cette dernière partie conserve les informations utiles pour comprendre, "
            "reproduire et documenter l'exécution du projet. Elle répond à la question : "
            "comment garder une trace du projet ?",
            "This final part keeps useful information for understanding, "
            "reproducing and documenting project execution. It answers the question: "
            "how to keep track of the project?"
        )
    )

    df, x_train, x_test, y_train, y_test = prepare_data()

    report_steps = [
        t("18. Créer les métadonnées", "18. Create metadata"),
        t("19. Enregistrer les fichiers JSON", "19. Save JSON files"),
        t("20. Afficher le résumé final", "20. Show final summary"),
    ]
    if "report_step" not in st.session_state:
        st.session_state.report_step = 0

    def go_to_report_step(step_index: int) -> None:
        st.session_state.report_step = step_index
        st.rerun()

    with st.container(horizontal=True):
        for index, step_name in enumerate(report_steps):
            st.button(
                f"Étape {index + 18}",
                key=f"report_nav_{index}",
                disabled=st.session_state.report_step == index,
                on_click=go_to_report_step if st.session_state.report_step != index else None,
                args=(index,) if st.session_state.report_step != index else (),
                width="stretch",
            )

    st.caption(f"{t('Étape actuelle', 'Current step')} : {report_steps[st.session_state.report_step]}")
    st.divider()

    if st.session_state.report_step == 0:
        st.header(t("18. Créer les métadonnées", "18. Create metadata"))
        st.write(
            t(
                "Les métadonnées sont une fiche d'identité de l'entraînement. Elles peuvent "
                "indiquer le modèle utilisé, les variables, les métriques, les versions des bibliothèques "
                "et le nombre de lignes utilisées.",
                "Metadata is an identity card for training. It can "
                "indicate the model used, variables, metrics, library versions "
                "and the number of rows used."
            )
        )
        metadata = {
            "projet": "Student Performance Predictor",
            "variables": list(x_train.columns),
            "lignes_entraînement": int(len(x_train)),
            "lignes_test": int(len(x_test)),
            "modèle": st.session_state.get("selected_name", t("non sélectionné", "not selected")),
            "métriques_test": st.session_state.get("test_metrics", {}),
        }
        st.json(metadata)
        st.session_state.metadata = metadata
        if st.button(t("Passer à l'enregistrement JSON →", "Go to JSON saving →"), type="primary", width="stretch"):
            go_to_report_step(1)

    elif st.session_state.report_step == 1:
        st.header(t("19. Enregistrer les fichiers JSON", "19. Save JSON files"))
        st.write(
            t(
                "Les informations techniques peuvent être enregistrées au format JSON. "
                "Ce format est lisible par Python et par de nombreux autres outils.",
                "Technical information can be saved in JSON format. "
                "This format is readable by Python and many other tools."
            )
        )
        metadata = st.session_state.get("metadata", {})
        if st.button(t("Enregistrer les métadonnées", "Save metadata"), type="primary", width="stretch"):
            metadata_path = Path("artifacts/metadata.json")
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            st.success(f"{t('Métadonnées enregistrées dans', 'Metadata saved in')} : {metadata_path}")
        with st.container(horizontal=True):
            if st.button(t("← Revenir aux métadonnées", "← Back to metadata"), width="stretch"):
                go_to_report_step(0)
            if st.button(t("Afficher le résumé final →", "Show final summary →"), type="primary", width="stretch"):
                go_to_report_step(2)

    else:
        st.header(t("20. Afficher le résumé final", "20. Show final summary"))
        st.write(
            t(
                "Le résumé final rappelle le modèle retenu, les performances observées et "
                "les fichiers produits par le projet.",
                "The final summary recalls the selected model, observed performances and "
                "files produced by the project."
            )
        )
        summary = {
            t("modèle sélectionné", "selected model"): st.session_state.get("selected_name", t("non disponible", "not available")),
            t("métriques de test", "test metrics"): st.session_state.get("test_metrics", {}),
            t("modèle prêt à être sauvegardé", "model ready to be saved"): st.session_state.get("production_model") is not None,
            t("rapport préparé", "report prepared"): True,
        }
        st.json(summary)
        st.success(t("La traçabilité du parcours est maintenant présentée.", "Walkthrough traceability is now presented."))
        
        st.markdown("---")
        st.subheader(t("🚀 Étape Ultime : Tester le modèle", "🚀 Ultimate Step: Test the model"))
        st.write(t(
            "Félicitations ! Le pipeline est terminé. Vous pouvez maintenant utiliser le fruit de ce travail "
            "pour prédire la performance d'un étudiant réel.",
            "Congratulations! The pipeline is complete. You can now use the fruit of this work "
            "to predict a real student's performance."
        ))
        
        if st.button(t("Accéder au Prédicateur de Performance →", "Access Performance Predictor →"), type="primary", width="stretch"):
            st.switch_page("app_pages/prediction.py")

        if st.button(t("Recommencer le rapport", "Restart report"), width="stretch"):
            go_to_report_step(0)

if __name__ == "__main__":
    final_report()
