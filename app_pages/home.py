import streamlit as st
from translations import t

def home_page():
    # Hero Section
    st.title("🎓 Student Data Mastery Pipeline")
    st.subheader(t(
        "L'Ingénierie de la Performance Scolaire par la Data Science",
        "Educational Performance Engineering through Data Science"
    ))
    
    st.markdown("---")
    
    # Value Proposition
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### 🎯 {t('Qu\'est-ce que c\'est ?', 'What is this?')}")
        st.write(t(
            "Ce projet n'est pas un simple prédicteur de notes. C'est un **cadre méthodologique complet** "
            "qui transforme des données brutes en décisions algorithmiques transparentes et auditables. "
            "Il simule l'intégralité du cycle de vie d'un projet de Machine Learning, de l'exploration à la production.",
            "This project is not a simple grade predictor. It is a **complete methodological framework** "
            "that transforms raw data into transparent and auditable algorithmic decisions. "
            "It simulates the entire Machine Learning project lifecycle, from exploration to production."
        ))
    with col2:
        st.image("Graph/logo_spp.webp", width=200)

    st.markdown("---")

    # Target Audience & Goals
    tab1, tab2, tab3 = st.tabs([
        t("🎯 Public Cible", "🎯 Target Audience"), 
        t("🚀 Objectifs", "🚀 Goals"), 
        t("🛠️ Méthodologie", "🛠️ Methodology")
    ])
    
    with tab1:
        st.markdown(t(
            "#### À qui s'adresse cet outil ?\n"
            "- **Analystes & Data Scientists Juniors** : Pour apprendre à structurer un pipeline rigoureux.\n"
            "- **Décideurs Académiques** : Pour comprendre les leviers de réussite des étudiants.\n"
            "- **Étudiants en IA** : Pour voir 'sous le capot' d'un modèle de régression complexe.",
            "#### Who is this tool for?\n"
            "- **Junior Data Scientists & Analysts**: To learn how to structure a rigorous pipeline.\n"
            "- **Academic Decision Makers**: To understand the drivers of student success.\n"
            "- **AI Students**: To see 'under the hood' of a complex regression model."
        ))

    with tab2:
        st.markdown(t(
            "#### Ce que vous allez accomplir :\n"
            "1. **Auditer** la qualité des données scolaires.\n"
            "2. **Explorer** visuellement les corrélations cachées.\n"
            "3. **Comparer** 5 algorithmes de pointe (Random Forest, Gradient Boosting, etc.).\n"
            "4. **Auditer** les erreurs et les biais pour une IA éthique.",
            "#### What you will achieve:\n"
            "1. **Audit** school data quality.\n"
            "2. **Explore** hidden correlations visually.\n"
            "3. **Compare** 5 state-of-the-art algorithms (Random Forest, Gradient Boosting, etc.).\n"
            "4. **Audit** errors and biases for ethical AI."
        ))
        
    with tab3:
        st.info(t(
            "Nous utilisons une approche **'Glass Box'** : chaque transformation, chaque métrique (RMSE, MAE, R²) "
            "et chaque importance de variable est expliquée et visualisée. "
            "Nous intégrons également **Pygwalker** pour une exploration interactive totale.",
            "We use a **'Glass Box'** approach: every transformation, every metric (RMSE, MAE, R²), "
            "and every feature importance is explained and visualized. "
            "We also integrate **Pygwalker** for total interactive exploration."
        ))

    st.markdown("---")

    # Call to Action
    st.success(t(
        "💡 **Prêt à commencer ?** Utilisez la barre latérale pour naviguer vers la 'Préparation des données'.",
        "💡 **Ready to start?** Use the sidebar to navigate to 'Data Preparation'."
    ))

    # Footer/Status
    st.divider()
    cols = st.columns(3)
    cols[0].metric(t("Statut", "Status"), "Production Ready")
    cols[1].metric(t("Modèles", "Models"), "5")
    cols[2].metric(t("Transparence", "Transparency"), "100%")

if __name__ == "__main__":
    home_page()
