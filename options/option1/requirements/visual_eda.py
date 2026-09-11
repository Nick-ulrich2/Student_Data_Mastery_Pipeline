import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from translations import t
from config import TARGET, NUMERIC_FEATURES

def visual_eda(df: pd.DataFrame):
    st.header(t("Analyse Visuelle Exploratoire (EDA)", "Visual Exploratory Data Analysis (EDA)"))
    st.write(
        t(
            "Un Data Scientist ne se contente pas de tableaux. Les graphiques révèlent "
            "les tendances cachées, les corrélations et les anomalies.",
            "A Data Scientist doesn't just look at tables. Charts reveal hidden trends, "
            "correlations, and anomalies."
        )
    )

    # 1. Heatmap de Corrélation
    st.subheader(t("Corrélation des variables numériques", "Numerical Features Correlation"))
    st.write(
        t(
            "Ce graphique montre comment les variables sont liées entre elles. Plus le carré est rouge "
            "ou bleu, plus le lien est fort. La corrélation avec la cible (Exam_Score) est cruciale.",
            "This chart shows how variables are linked. The redder or bluer the square, the stronger "
            "the link. Correlation with the target (Exam_Score) is crucial."
        )
    )
    
    numeric_df = df[NUMERIC_FEATURES + [TARGET]].select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    # 2. Distribution de la cible
    st.subheader(t("Analyse de la Distribution des Notes", "Score Distribution Analysis"))
    st.write(
        t(
            "Si la distribution est 'normale' (en forme de cloche), le modèle aura plus de facilité "
            "à prédire les valeurs moyennes.",
            "If the distribution is 'normal' (bell-shaped), the model will find it easier to "
            "predict average values."
        )
    )
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(df[TARGET], kde=True, color='skyblue', ax=ax2)
    ax2.set_title(t(f"Distribution de {TARGET}", f"{TARGET} Distribution"))
    st.pyplot(fig2)

    # 3. Boxplots pour les catégories
    st.subheader(t("Impact des catégories sur la performance", "Category Impact on Performance"))
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        selected_cat = st.selectbox(t("Choisir une catégorie pour voir l'impact", "Choose a category to see impact"), cat_cols)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=selected_cat, y=TARGET, data=df, ax=ax3, palette='Set2')
        plt.xticks(rotation=45)
        st.pyplot(fig3)
