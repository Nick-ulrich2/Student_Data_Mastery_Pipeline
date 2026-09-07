import pandas as pd
import streamlit as st

from translations import t


def data_splitting(df, x_train, x_test):
    st.header(t("4. Division des données", "4. Data split"))
    st.write(
        t(
            "Le dataset est divisé en deux parties. Le jeu d'entraînement sert au modèle pour apprendre. Le jeu de test sert ensuite à vérifier si le modèle fonctionne sur des données qu'il n'a jamais vues.",
            "The dataset is split into two parts. The training set is used by the model to learn. The test set then checks how the model performs on unseen data.",
        )
    )
    st.info(
        t(
            "Cette séparation permet d'éviter de croire qu'un modèle est performant simplement parce qu'il a mémorisé les données utilisées pendant son apprentissage.",
            "This split prevents us from assuming a model is effective simply because it memorized its training data.",
        )
    )

    train_df = df.loc[x_train.index].copy()
    test_df = df.loc[x_test.index].copy()

    split_col1, split_col2 = st.columns(2)
    split_col1.metric(
        t("Données d'entraînement", "Training data"),
        f"{len(train_df)} lignes",
        f"{len(train_df) / len(df) * 100:.1f} % du dataset",
    )
    split_col2.metric(
        t("Données de test", "Test data"),
        f"{len(test_df)} lignes",
        f"{len(test_df) / len(df) * 100:.1f} % du dataset",
    )

    st.subheader(t("Exemple de données d'entraînement", "Training data example"))
    st.write(
        t(
            "Ces lignes sont utilisées pour apprendre les relations entre les caractéristiques des étudiants et leur note.",
            "These rows are used to learn relationships between student characteristics and their score.",
        )
    )
    st.dataframe(train_df.head(5), width="stretch", hide_index=True)

    st.subheader(t("Exemple de données de test", "Test data example"))
    st.write(
        t(
            "Ces lignes sont conservées à part. Elles servent à évaluer la capacité du modèle à produire de bonnes prédictions sur de nouveaux étudiants.",
            "These rows are kept separate to evaluate the model's ability to make good predictions for new students.",
        )
    )
    st.dataframe(test_df.head(5), width="stretch", hide_index=True)

    st.caption(
        t(
            "Attention : l'exemple présenté ici utilise la séparation déjà produite par prepare_data(), afin de rester cohérent avec le modèle.",
            "Note: this example uses the split already produced by prepare_data() to remain consistent with the model.",
        )
    )
    st.divider()
