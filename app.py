import streamlit as st
from themes import apply_theme, set_theme
from translations import LANGUAGES, install_ui_translation, set_language, t

# Page configurations
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="Graph/logo_spp.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)

install_ui_translation()

# Global styles
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

# Sidebar UI
st.sidebar.image("Graph/logo_spp.webp", width=200)

# Language management
previous_language = st.session_state.get("language", "en")
language_label = st.sidebar.selectbox(
    t("Langue", "Language"),
    list(LANGUAGES),
    index=1 if st.session_state.get("language", "en") == "en" else 0,
    key="language_choice",
)
selected_language = LANGUAGES[language_label]
set_language(selected_language)
if selected_language != previous_language:
    st.rerun()

# Theme management
previous_theme = st.session_state.get("theme", "dark")
theme_value = st.sidebar.radio(
    t("Thème", "Theme"),
    ["light", "dark"],
    index=1 if previous_theme == "dark" else 0,
    format_func=lambda value: t("Clair", "Light") if value == "light" else t("Sombre", "Dark"),
    key="theme_choice",
)
selected_theme = "dark" if theme_value in {"dark", "Dark", "Sombre"} else "light"
set_theme(selected_theme)
apply_theme(selected_theme)

# Navigation definition
from app_pages.home import home_page
from app_pages.data_preparation import data_preparation
from app_pages.model_training import model_training
from app_pages.model_evaluation import model_evaluation
from app_pages.final_results import final_results
from app_pages.final_report import final_report
from app_pages.prediction import prediction_page

pages = [
    st.Page(
        home_page,
        title=t("Accueil", "Home"),
        icon=":material/home:",
        url_path="home",
        default=True
    ),
    st.Page(
        data_preparation, 
        title=t("Préparation des données", "Data preparation"), 
        icon=":material/database:", 
        url_path="data_preparation",
    ),
    st.Page(
        model_training, 
        title=t("Entraînement du modèle", "Model training"), 
        icon=":material/model_training:", 
        url_path="model_training"
    ),
    st.Page(
        model_evaluation, 
        title=t("Évaluation du modèle", "Model evaluation"), 
        icon=":material/query_stats:", 
        url_path="model_evaluation"
    ),
    st.Page(
        final_results, 
        title=t("Résultats finaux", "Final results"), 
        icon=":material/analytics:", 
        url_path="final_results"
    ),
    st.Page(
        prediction_page,
        title=t("Mode Prédicteur", "Predictor Mode"),
        icon=":material/rocket_launch:",
        url_path="predictor"
    ),
    st.Page(
        final_report, 
        title=t("Rapport et traçabilité", "Report and traceability"), 
        icon=":material/description:", 
        url_path="final_report"
    ),
]

# Run navigation
pg = st.navigation(pages)
pg.run()
