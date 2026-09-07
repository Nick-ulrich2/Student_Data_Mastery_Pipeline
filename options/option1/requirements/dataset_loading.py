import streamlit as st
from translations import t

def dataset_loading(df):
    st.header(t("1. Chargement des données", "1. Loading the data"))
    st.write(
        t(
            "Cet onglet présente les données telles qu'elles ont été chargées. Chaque ligne représente généralement un étudiant et chaque colonne représente une information sur cet étudiant.",
            "This section presents the data as it was loaded. Each row generally represents a student and each column represents information about that student.",
        )
    )
    st.markdown(
        t(
            "**Comment lire ce tableau ?** Les colonnes correspondent aux caractéristiques utilisées pour comprendre ou prédire la note. La colonne de résultat, par exemple `Exam_Score`, représente la variable que le modèle essaie de prédire.",
            "**How should this table be read?** The columns are the features used to understand or predict the score. The result column, such as `Exam_Score`, is the variable the model tries to predict.",
        )
    )
    st.dataframe(df, width="stretch", hide_index=True)

    st.divider()
