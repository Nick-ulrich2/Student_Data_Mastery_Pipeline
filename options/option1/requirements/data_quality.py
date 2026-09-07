import streamlit as st
import pandas as pd
from translations import t

def get_data_quality_indicators(df):
    st.header(t("3. Qualité des données", "3. Data quality"))
    st.write(t(
            "La qualité des données indique si le fichier contient des informations manquantes, des lignes répétées ou des valeurs potentiellement difficiles à utiliser par le modèle.",
            "Data quality indicates whether the file contains missing information, duplicate rows or values that may be difficult for the model to use."
    ))
    quality_table = pd.DataFrame(
        {
            t("Colonne", "Column"): df.columns,
            t("Valeurs manquantes", "Missing values"): df.isna().sum().values,
            t("Pourcentage manquant", "Missing percentage"): df.isna().mean().mul(100).round(2).values,
            t("Valeurs uniques", "Unique values"): [
                df[column].nunique(dropna=True) for column in df.columns
            ],
        }
    )
    
    missing_values = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    st.subheader(t("Valeurs manquantes par colonne", "Missing values by column"))
    st.dataframe(quality_table, width="stretch", hide_index=True)
    if missing_values == 0:
            st.success(t("Aucune valeur manquante n'a été détectée dans le dataset.", "No missing value was found in the dataset."))
    else:
        st.warning(
                t(
                    f"{missing_values} valeur(s) manquante(s) ont été détectées. Elles devront être traitées avant ou pendant l'entraînement.",
                    f"{missing_values} missing value(s) were found. They must be handled before or during training."
                )
        )
        st.subheader(t("Lignes dupliquées", "Duplicate rows"))
    if duplicate_rows == 0:
        st.success(t("Aucune ligne dupliquée n'a été détectée.", "No duplicate row was found."))
    else:
        st.warning(
                t(
                    f"{duplicate_rows} ligne(s) dupliquée(s) ont été détectées. Elles peuvent influencer les statistiques et l'apprentissage.",
                    "They can affect statistics and model training."
                )
        )
        st.subheader(t("Interprétation", "Interpretation"))
    st.write(
        t(
            "Une donnée manquante ne signifie pas forcément que le projet doit être arrêté. Elle peut être remplacée par une valeur représentative, comme la médiane pour une variable numérique ou la valeur la plus fréquente pour une variable catégorielle.",
            "A missing value does not necessarily mean the project must stop. It can be replaced with a representative value, such as the median for a numeric variable or the most frequent value for a categorical variable."
        )
    )

    st.divider()