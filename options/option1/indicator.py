import streamlit as st
from translations import t

def get_general_indicators(df):
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    missing_values = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    metric_col1.metric(t("Nombre d'étudiants", "Number of students"), total_rows),
    metric_col2.metric(t("Nombre de variables", "Number of variables"), total_columns),
    metric_col3.metric(t("Valeurs manquantes", "Missing values"), missing_values),
    metric_col4.metric(t("Lignes dupliquées", "Duplicate rows"), duplicate_rows)
