import streamlit as st
import pandas as pd
from translations import t

def dataset_struct(df: pd.DataFrame) -> None:
    st.header(t("2. Structure des données", "2. Data structure"))
    st.write(
        t(
            "Cet onglet permet de comprendre le type et la répartition des variables. Une variable numérique contient des nombres, tandis qu'une variable catégorielle contient des catégories ou du texte.",
            "This section explains the type and distribution of the variables. A numeric variable contains numbers, while a categorical variable contains categories or text.",
        )
    )
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
    structure_col1, structure_col2 = st.columns(2)
    structure_col1.metric(t("Variables numériques", "Numeric variables"), len(numeric_columns))
    structure_col2.metric(t("Variables catégorielles", "Categorical variables"), len(categorical_columns))
    st.subheader(t("Types des colonnes", "Column types"))
    dtype_table = pd.DataFrame(
        {
            t("Colonne", "Column"): df.columns,
            t("Type détecté", "Detected type"): df.dtypes.astype(str).values,
            t("Nombre de valeurs différentes", "Number of distinct values"): [
                df[column].nunique(dropna=True) for column in df.columns
            ],
        }
    )
    st.dataframe(dtype_table, width="stretch", hide_index=True)
    st.subheader(t("Résumé statistique", "Statistical summary"))
    st.write(
        t(
            "Pour les variables numériques, ce résumé montre notamment la moyenne, la valeur minimale, la valeur maximale et les quartiles. Ces informations aident à repérer des valeurs inhabituelles.",
            "For numeric variables, this summary shows the mean, minimum, maximum and quartiles. This information helps identify unusual values.",
        )
    )
    st.dataframe(
        df.describe(include="all").transpose(),
        width="stretch",
    )