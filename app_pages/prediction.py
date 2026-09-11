import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from translations import t
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

def prediction_page():
    st.title(t("🚀 Mode Production : Prédicteur", "🚀 Production Mode: Predictor"))
    
    model_path = Path("artifacts/models/model.joblib")
    
    if not model_path.exists():
        st.warning(t(
            "Le modèle final n'a pas encore été entraîné et sauvegardé. Veuillez terminer le pipeline d'abord.",
            "The final model has not been trained and saved yet. Please complete the pipeline first."
        ))
        return

    st.info(t(
        "Saisissez les caractéristiques d'un étudiant pour estimer sa performance à l'examen.",
        "Enter student characteristics to estimate exam performance."
    ))

    # Formulaire de saisie sécurisé
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        inputs = {}
        
        with col1:
            st.markdown(f"### {t('Variables Numériques', 'Numerical Variables')}")
            inputs['Hours_Studied'] = st.number_input(t("Heures d'étude", "Hours Studied"), 0, 168, 20)
            inputs['Attendance'] = st.number_input(t("Présence (%)", "Attendance (%)"), 0, 100, 80)
            inputs['Sleep_Hours'] = st.number_input(t("Heures de sommeil", "Sleep Hours"), 0, 24, 8)
            inputs['Previous_Scores'] = st.number_input(t("Scores précédents", "Previous Scores"), 0, 100, 70)
            inputs['Tutoring_Sessions'] = st.number_input(t("Sessions de tutorat", "Tutoring Sessions"), 0, 50, 2)
            inputs['Physical_Activity'] = st.number_input(t("Activité physique (h/sem)", "Physical Activity (h/wk)"), 0, 50, 5)
            inputs['Class_Size'] = st.number_input(t("Taille de la classe", "Class Size"), 1, 100, 25)

        with col2:
            st.markdown(f"### {t('Variables Catégorielles', 'Categorical Variables')}")
            inputs['Parental_Involvement'] = st.selectbox(t("Implication parentale", "Parental Involvement"), ["Low", "Medium", "High"])
            inputs['Access_to_Resources'] = st.selectbox(t("Accès aux ressources", "Access to Resources"), ["Low", "Medium", "High"])
            inputs['Extracurricular_Activities'] = st.selectbox(t("Activités extrascolaires", "Extracurricular Activities"), ["Yes", "No"])
            inputs['Internet_Access'] = st.selectbox(t("Accès Internet", "Internet Access"), ["Yes", "No"])
            inputs['Family_Income'] = st.selectbox(t("Revenu familial", "Family Income"), ["Low", "Medium", "High"])
            inputs['School_Type'] = st.selectbox(t("Type d'école", "School Type"), ["Public", "Private"])
            inputs['Parental_Education_Level'] = st.selectbox(t("Niveau d'éducation des parents", "Parental Education Level"), ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Postgraduate"])
            inputs['Distance_from_Home'] = st.selectbox(t("Distance de la maison", "Distance from Home"), ["Near", "Moderate", "Far"])
            inputs['Gender'] = st.selectbox(t("Genre", "Gender"), ["Male", "Female"])
            inputs['Electricity_Access'] = st.selectbox(t("Accès à l'électricité", "Electricity Access"), ["Yes", "No"])
            inputs['Region'] = st.selectbox(t("Région", "Region"), ["Urban", "Suburban", "Rural"])
            inputs['Transport_Mode'] = st.selectbox(t("Mode de transport", "Transport Mode"), ["Walking", "Public Transport", "Private", "School Bus"])
            inputs['Language_Section'] = st.selectbox(t("Section linguistique", "Language Section"), ["French", "English", "Other"])

        submit = st.form_submit_button(t("Prédire la performance", "Predict performance"), type="primary", width="stretch")

    if submit:
        try:
            # Sécurité : Chargement du modèle avec cache
            model = joblib.load(model_path)
            
            # Transformation des inputs en DataFrame
            input_df = pd.DataFrame([inputs])
            
            # Prédiction
            prediction = model.predict(input_df)[0]
            
            # Post-processing (Sanitization du résultat)
            prediction = max(0, min(100, prediction))
            
            st.balloons()
            st.metric(t("Score d'examen estimé", "Estimated Exam Score"), f"{prediction:.2f} / 100")
            
            st.warning(t(
                "⚠️ Avertissement : Cette prédiction est basée sur des corrélations statistiques. "
                "Elle ne prend pas en compte les facteurs psychologiques ou contextuels imprévus.",
                "⚠️ Warning: This prediction is based on statistical correlations. "
                "It does not account for unforeseen psychological or contextual factors."
            ))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

if __name__ == "__main__":
    prediction_page()
