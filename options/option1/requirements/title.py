import streamlit as st
from translations import t

def title():
    st.title(t("Préparation et compréhension des données", "Data preparation and understanding"))
    st.info(
        t(
            "Pourquoi cette étape est-elle importante ? Une prédiction n'est fiable "
            "que si les données sont de bonne qualité. Il est donc important de "
            "comprendre les données avant de les utiliser pour entraîner un modèle.",
            "Why is this step important? A prediction is reliable only when the data "
            "is high quality. Understanding the data before training a model is therefore essential.",
        )
    )