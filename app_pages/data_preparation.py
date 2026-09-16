import streamlit as st
from options.option1.indicator import get_general_indicators
from options.option1.requirements.data_splitting import data_splitting
from options.option1.requirements.data_quality import get_data_quality_indicators
from options.option1.requirements.dataset_loading import dataset_loading
from options.option1.requirements.dataset_struct import dataset_struct
from options.option1.requirements.nav_buttons import nav_buttons, go_to_step
from options.option1.requirements.title import title
from train import prepare_data
from translations import t

from options.option1.requirements.visual_eda import visual_eda

def data_preparation():
    
    # Affichage du titre de la page
    title()

    # chargement des données
    df, x_train, x_test, y_train, y_test = prepare_data()

    """Calcule les indicateurs généraux du dataset."""
    get_general_indicators(df)

    st.divider()

    # Navigation entre les étapes
    step_names = [
        t("1. Que contiennent les données ?", "1. What do the data contain?"),
        t("2. Comment sont-elles organisées ?", "2. How are they organized?"),
        t("3. Peut-on leur faire confiance ?", "3. Can they be trusted?"),
        t("4. Analyse visuelle (EDA)", "4. Visual analysis (EDA)"),
        t("5. Comment sont-elles séparées ?", "5. How are they split?"),
    ]

    # Gestion de l'état de l'étape actuelle
    if "data_step" not in st.session_state:
        st.session_state.data_step = 0

    # Affichage des boutons de navigation pour chaque étape
    with st.container(horizontal=True):
        nav_buttons(step_names, go_to_step)

    st.caption(
        f"{t('Étape actuelle', 'Current step')}: {st.session_state.data_step + 1} / "
        f"{len(step_names)} — {step_names[st.session_state.data_step]}"
    )
    st.divider()

    # Étape 1 : chargement des données
    if st.session_state.data_step == 0:
        dataset_loading(df)
        
        if st.button(
            t("Passer à la structure des données →", "Go to data structure →"),
            key="next_to_structure",
            type="primary",
            width="stretch",
        ):
            go_to_step(1)

    # Étape 2 : structure des données
    elif st.session_state.data_step == 1:
        dataset_struct(df)

        st.divider()
        with st.container(horizontal=True):
            if st.button(
                t("← Revenir au chargement", "← Back to loading"),
                key="back_to_load",
                width="stretch",
            ):
                go_to_step(0)
            if st.button(
                t("Passer à la qualité des données →", "Go to data quality →"),
                key="next_to_quality",
                type="primary",
                width="stretch",
            ):
                go_to_step(2)

    # Étape 3 : qualité des données
    elif st.session_state.data_step == 2:
        get_data_quality_indicators(df)
        
        with st.container(horizontal=True):
            if st.button(
                t("← Revenir à la structure", "← Back to structure"),
                key="back_to_structure",
                width="stretch",
            ):
                go_to_step(1)
            if st.button(
                t("Passer à l'analyse visuelle →", "Go to visual analysis →"),
                key="next_to_eda",
                type="primary",
                width="stretch",
            ):
                go_to_step(3)

    # Étape 4 : EDA Visuelle
    elif st.session_state.data_step == 3:
        visual_eda(df)
        
        with st.container(horizontal=True):
            if st.button(
                t("← Revenir à la qualité", "← Back to quality"),
                key="back_to_quality_from_eda",
                width="stretch",
            ):
                go_to_step(2)
            if st.button(
                t("Passer à l'exploration interactive →", "Go to interactive exploration →"),
                key="next_to_pyg",
                type="primary",
                width="stretch",
            ):
                go_to_step(4)

    # Étape 5 : division des données
    elif st.session_state.data_step == 4:
        data_splitting(df, x_train, x_test)
        
        with st.container(horizontal=True):
            if st.button(
                t("← Revenir à l'analyse visuelle", "← Back to visual analysis"),
                key="back_to_eda_from_split",
                width="stretch",
            ):
                go_to_step(3)
            if st.button(
                t("Recommencer le parcours", "Restart the walkthrough"),
                key="restart_data_steps",
                type="primary",
                width="stretch",
            ):
                go_to_step(0)

if __name__ == "__main__":
    data_preparation()



# # Étape 2 : structure des données
# elif st.session_state.data_step == 1:
#     st.header("2. Structure des données")
#     st.write(
#         "Cet onglet permet de comprendre le type et la répartition des variables. "
#         "Une variable numérique contient des nombres, tandis qu'une variable "
#         "catégorielle contient des catégories ou du texte."
#     )
#     numeric_columns = df.select_dtypes(include="number").columns.tolist()
#     categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
#     structure_col1, structure_col2 = st.columns(2)
#     structure_col1.metric("Variables numériques", len(numeric_columns))
#     structure_col2.metric("Variables catégorielles", len(categorical_columns))
#     st.subheader("Types des colonnes")
#     dtype_table = pd.DataFrame(
#         {
#             "Colonne": df.columns,
#             "Type détecté": df.dtypes.astype(str).values,
#             "Nombre de valeurs différentes": [
#                 df[column].nunique(dropna=True) for column in df.columns
#             ],
#         }
#     )
#     st.dataframe(dtype_table, width="stretch", hide_index=True)
#     st.subheader("Résumé statistique")
#     st.write(
#         "Pour les variables numériques, ce résumé montre notamment la moyenne, "
#         "la valeur minimale, la valeur maximale et les quartiles. Ces informations "
#         "aident à repérer des valeurs inhabituelles."
#     )
#     st.dataframe(
#         df.describe(include="all").transpose(),
#         width="stretch",
#     )

#     st.divider()
#     previous_col, next_col = st.columns(2)
#     with previous_col:
#         if st.button(
#             "← Revenir au chargement",
#             key="back_to_load",
#             width="stretch",
#         ):
#             go_to_step(0)
#     with next_col:
#         if st.button(
#             "Passer à la qualité des données →",
#             key="next_to_quality",
#             type="primary",
#             width="stretch",
#         ):
#             go_to_step(2)

# # Étape 3 : qualité des données
# elif st.session_state.data_step == 2:
#     st.header("3. Qualité des données")
#     st.write(
#         "La qualité des données indique si le fichier contient des informations "
#         "manquantes, des lignes répétées ou des valeurs potentiellement difficiles "
#         "à utiliser par le modèle."
#     )
#     quality_table = pd.DataFrame(
#         {
#             "Colonne": df.columns,
#             "Valeurs manquantes": df.isna().sum().values,
#             "Pourcentage manquant": df.isna().mean().mul(100).round(2).values,
#             "Valeurs uniques": [
#                 df[column].nunique(dropna=True) for column in df.columns
#             ],
#         }
#     )
#     st.subheader("Valeurs manquantes par colonne")
#     st.dataframe(quality_table, width="stretch", hide_index=True)
#     if missing_values == 0:
#         st.success("Aucune valeur manquante n'a été détectée dans le dataset.")
#     else:
#         st.warning(
#             f"{missing_values} valeur(s) manquante(s) ont été détectées. "
#             "Elles devront être traitées avant ou pendant l'entraînement."
#         )
#     st.subheader("Lignes dupliquées")
#     if duplicate_rows == 0:
#         st.success("Aucune ligne dupliquée n'a été détectée.")
#     else:
#         st.warning(
#             f"{duplicate_rows} ligne(s) dupliquée(s) ont été détectées. "
#             "Elles peuvent influencer les statistiques et l'apprentissage."
#         )
#     st.subheader("Interprétation")
#     st.write(
#         "Une donnée manquante ne signifie pas forcément que le projet doit être "
#         "arrêté. Elle peut être remplacée par une valeur représentative, comme "
#         "la médiane pour une variable numérique ou la valeur la plus fréquente "
#         "pour une variable catégorielle."
#     )

#     st.divider()
#     previous_col, next_col = st.columns(2)
#     with previous_col:
#         if st.button(
#             "← Revenir à la structure",
#             key="back_to_structure",
#             width="stretch",
#         ):
#             go_to_step(1)
#     with next_col:
#         if st.button(
#             "Passer à la division des données →",
#             key="next_to_split",
#             type="primary",
#             width="stretch",
#         ):
#             go_to_step(3)

# # Étape 4 : division des données
# elif st.session_state.data_step == 3:
#     st.header("4. Division des données")
#     st.write(
#         "Le dataset est divisé en deux parties. Le jeu d'entraînement sert au "
#         "modèle pour apprendre. Le jeu de test sert ensuite à vérifier si le "
#         "modèle fonctionne sur des données qu'il n'a jamais vues."
#     )
#     st.info(
#         "Cette séparation permet d'éviter de croire qu'un modèle est performant "
#         "simplement parce qu'il a mémorisé les données utilisées pendant son apprentissage."
#     )

#     train_df = df.loc[x_train.index].copy()
#     test_df = df.loc[x_test.index].copy()

#     split_col1, split_col2 = st.columns(2)
#     split_col1.metric(
#         "Données d'entraînement",
#         f"{len(train_df)} lignes",
#         f"{len(train_df) / len(df) * 100:.1f} % du dataset",
#     )
#     split_col2.metric(
#         "Données de test",
#         f"{len(test_df)} lignes",
#         f"{len(test_df) / len(df) * 100:.1f} % du dataset",
#     )

#     st.subheader("Exemple de données d'entraînement")
#     st.write(
#         "Ces lignes sont utilisées pour apprendre les relations entre les "
#         "caractéristiques des étudiants et leur note."
#     )
#     st.dataframe(train_df.head(5), width="stretch", hide_index=True)

#     st.subheader("Exemple de données de test")
#     st.write(
#         "Ces lignes sont conservées à part. Elles servent à évaluer la capacité "
#         "du modèle à produire de bonnes prédictions sur de nouveaux étudiants."
#     )
#     st.dataframe(test_df.head(5), width="stretch", hide_index=True)

#     st.caption(
#         "Attention : l'exemple présenté ici utilise la séparation déjà produite "
#         "par prepare_data(), afin de rester cohérent avec le modèle."
#     )

#     st.divider()
#     previous_col, finish_col = st.columns(2)
#     with previous_col:
#         if st.button(
#             "← Revenir à la qualité",
#             key="back_to_quality",
#             width="stretch",
#         ):
#             go_to_step(2)
#     with finish_col:
#         if st.button(
#             "Recommencer le parcours",
#             key="restart_data_steps",
#             type="primary",
#             width="stretch",
#         ):
#             go_to_step(0)