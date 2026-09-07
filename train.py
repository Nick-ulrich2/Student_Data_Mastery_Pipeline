from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import *
from themes import configure_matplotlib_theme, theme_chart_colors
from translations import t

sns.set_theme(style="whitegrid", context="notebook")

@st.cache_data
def load_data():
    return pd.read_csv("data/SPP.csv")

def validate_data(data : pd.DataFrame, TARGET) -> pd.DataFrame:
    # Vu que le max est a 101, je le remets a 100
    data.loc[data[TARGET] > 100, TARGET] = 100
    
    x = data.drop(TARGET, axis=1)
    y = data[TARGET]
    
    return x, y
    
def prepare_data():
    df = load_data()
    x, y = validate_data(df, TARGET)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return df, x_train, x_test, y_train, y_test

def make_preprocessor() -> ColumnTransformer:
    """Prépare les colonnes numériques et catégorielles avant l'entraînement."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop", # Toutes les colonnes qui ne sont pas présentes dans NUMERIC_FEATURES ou CATEGORICAL_FEATURES sont supprimées.
        verbose_feature_names_out=True, # Conserve les colonnes non déclarées sans les transformer
    )


def make_pipeline(model: Any) -> Pipeline:
    """Construit une chaîne composée de la préparation puis du modèle."""
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", model),
        ]
    )


def model_candidates() -> dict[str, Pipeline]:
    """Retourne les modèles qui seront comparés."""
    return {
        "dummy_mean": make_pipeline(DummyRegressor(strategy="mean")),
        "ridge": make_pipeline(Ridge(alpha=10.0)),
        "random_forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "extra_trees": make_pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=8,
                loss="huber",
                random_state=RANDOM_STATE,
            )
        ),
    }


def metric_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les trois métriques principales de régression."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# def dataset_manifest(path: Path) -> dict[str, object]:
#     """Crée une fiche d'identité du fichier de données."""
#     raw_bytes = path.read_bytes()
#     frame = pd.read_csv(path)
#     return {
#         "file_name": path.name,
#         "sha256": hashlib.sha256(raw_bytes).hexdigest(),
#         "bytes": len(raw_bytes),
#         "rows": int(frame.shape[0]),
#         "columns": int(frame.shape[1]),
#     }


def cross_validate_candidates(
    candidates: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    show_streamlit: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Compare les modèles par validation croisée et affiche les résultats dans Streamlit."""
    if CV_FOLDS < 2:
        raise ValueError(t("CV_FOLDS doit être au moins égal à 2.", "CV_FOLDS must be at least 2."))
    if len(x_train) < CV_FOLDS:
        raise ValueError(t(
            "Le nombre de lignes d'entraînement doit être supérieur à CV_FOLDS.",
            "The number of training rows must be greater than CV_FOLDS.",
        ))

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    cv_details: dict[str, dict[str, float]] = {}
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    progress = st.progress(0.0) if show_streamlit else None
    total = len(candidates)

    for position, (name, pipeline) in enumerate(candidates.items(), start=1):
        result = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )

        rmse_values = -result["test_rmse"]
        mae_values = -result["test_mae"]
        r2_values = result["test_r2"]
        row = {
            "model": name,
            "cv_mae_mean": float(mae_values.mean()),
            "cv_mae_std": float(mae_values.std(ddof=1)),
            "cv_rmse_mean": float(rmse_values.mean()),
            "cv_rmse_std": float(rmse_values.std(ddof=1)),
            "cv_r2_mean": float(r2_values.mean()),
            "cv_r2_std": float(r2_values.std(ddof=1)),
        }
        rows.append(row)
        cv_details[name] = {
            key: float(value) for key, value in row.items() if key != "model"
        }

        if show_streamlit:
            st.write(f"{t('Résultats du modèle', 'Model results')} **{name}**")
            st.dataframe(pd.DataFrame([row]), width="stretch", hide_index=True)
            progress.progress(position / total)

    comparison = (
        pd.DataFrame(rows)
        .sort_values("cv_rmse_mean", ascending=True)
        .reset_index(drop=True)
    )

    if show_streamlit:
        st.subheader(t("Comparaison finale des modèles", "Final model comparison"))
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.success(
            f"{t('Meilleur modèle selon le RMSE :', 'Best model according to RMSE:')} "
            f"{comparison.iloc[0]['model']}"
        )

    return comparison, cv_details


def aggregate_importance(
    pipeline: Pipeline,
    show_streamlit: bool = True,
    top_n: int = 12,
) -> pd.DataFrame:
    """Agrège l'importance des colonnes transformées et l'affiche dans Streamlit."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        direction = "importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        direction = "absolute_coefficient"
    else:
        empty = pd.DataFrame(columns=["source", "kind", "importance"])
        if show_streamlit:
            st.info(t("Ce modèle ne fournit pas d'importance de variables.", "This model does not provide feature importance."))
        return empty

    if len(feature_names) != len(values):
        raise ValueError(t(
            "Le nombre de noms de variables transformées ne correspond pas au nombre d'importances.",
            "The number of transformed feature names does not match the number of importances.",
        ))

    all_source_columns = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    rows: list[dict[str, object]] = []
    for name, value in zip(feature_names, values):
        source = name
        for column in all_source_columns:
            if name == f"num__{column}" or name.startswith(f"cat__{column}_"):
                source = column
                break
        rows.append(
            {
                "feature": name,
                "importance": float(value),
                "source": source,
                "kind": direction,
            }
        )

    detailed = pd.DataFrame(rows)
    aggregated = (
        detailed.groupby(["source", "kind"], as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if show_streamlit and not aggregated.empty:
        st.subheader(t("Variables les plus importantes", "Most important variables"))
        top = aggregated.head(top_n).sort_values("importance", ascending=True)
        chart_data = top.set_index("source")["importance"]
        st.bar_chart(chart_data)
        st.dataframe(aggregated.head(top_n), width="stretch", hide_index=True)

    return aggregated


def subgroup_metrics(test_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Mesure les performances du modèle pour chaque sous-groupe."""
    missing_columns = [column for column in GROUP_COLUMNS if column not in test_df.columns]
    if missing_columns:
        raise KeyError(t(
            f"Colonnes de groupes absentes des données : {missing_columns}",
            f"Group columns missing from the data: {missing_columns}",
        ))

    frame = test_df.copy()
    frame["prediction"] = predictions
    frame["absolute_error"] = (frame[TARGET] - frame["prediction"]).abs()
    frame["squared_error"] = (frame[TARGET] - frame["prediction"]) ** 2
    rows: list[dict[str, object]] = []

    for column in GROUP_COLUMNS:
        for group, subset in frame.groupby(column, dropna=False):
            rows.append(
                {
                    "group_column": column,
                    "group": "Missing" if pd.isna(group) else str(group),
                    "n": int(len(subset)),
                    "mae": float(subset["absolute_error"].mean()),
                    "rmse": float(np.sqrt(subset["squared_error"].mean())),
                    "mean_actual": float(subset[TARGET].mean()),
                    "mean_prediction": float(subset["prediction"].mean()),
                }
            )
    return pd.DataFrame(rows)


def save_figures(
    train_df: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    importances: pd.DataFrame,
    show_streamlit: bool = True,
) -> None:
    """Crée, enregistre et affiche les graphiques dans Streamlit."""
    configure_matplotlib_theme()
    chart_colors = theme_chart_colors()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, plt.Figure]] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(train_df[TARGET], bins=30, kde=True, color=chart_colors["blue"], ax=ax)
    ax.set_title(t(f"Distribution de {TARGET}", f"{TARGET} distribution"))
    ax.set_xlabel(TARGET)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=160)
    figures.append((t("Distribution des notes", "Score distribution"), fig))

    residuals = y_test.to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.55, color=chart_colors["orange"], ax=ax)
    ax.axhline(0, color=chart_colors["text"], linewidth=1)
    ax.set_title(t("Résidus sur le jeu de test", "Residuals on the test set"))
    ax.set_xlabel(t("Prédiction", "Prediction"))
    ax.set_ylabel(t("Résidu réel - prédit", "Actual minus predicted residual"))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals.png", dpi=160)
    figures.append((t("Résidus", "Residuals"), fig))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.55, color=chart_colors["green"], ax=ax)
    limits = [
        min(float(y_test.min()), float(np.min(predictions))),
        max(float(y_test.max()), float(np.max(predictions))),
    ]
    ax.plot(limits, limits, linestyle="--", color=chart_colors["text"])
    ax.set_title(t("Scores réels et prédits", "Actual and predicted scores"))
    ax.set_xlabel(t("Score réel", "Actual score"))
    ax.set_ylabel(t("Score prédit", "Predicted score"))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=160)
    figures.append((t("Scores réels et prédits", "Actual and predicted scores"), fig))

    if not importances.empty:
        top = importances.head(12).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=top, x="importance", y="source", color=chart_colors["blue"], ax=ax)
        ax.set_title(t("Variables les plus importantes pour le modèle retenu", "Most important variables for the selected model"))
        ax.set_xlabel(t("Importance agrégée", "Aggregated importance"))
        ax.set_ylabel(t("Variable source", "Source variable"))
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
        figures.append((t("Importance des variables", "Feature importance"), fig))

    if show_streamlit:
        st.subheader(t("Graphiques d'analyse", "Analysis charts"))
        for title, figure in figures:
            st.write(f"**{title}**")
            st.pyplot(figure, clear_figure=False)

    for _, figure in figures:
        plt.close(figure)


def main() -> None:
    """Exécute l'entraînement complet du projet."""

    df = load_data()
    
    x, y = validate_data(df, TARGET)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    
    train_df = df.loc[x_train.index].copy()
    test_df = df.loc[x_test.index].copy()

    candidates = model_candidates()
    comparison, cv_details = cross_validate_candidates(candidates, x_train, y_train)
    selected_name = str(comparison.iloc[0]["model"])
    selected_pipeline = candidates[selected_name]
    selected_pipeline.fit(x_train, y_train)

    test_predictions = selected_pipeline.predict(x_test)
    test_metrics = metric_dict(y_test, test_predictions)
    importances = aggregate_importance(selected_pipeline)
    subgroup = subgroup_metrics(test_df, test_predictions)

    importances.to_csv(IMPORTANCE_PATH, index=False)
    subgroup.to_csv(SUBGROUP_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)

    test_output = test_df[[TARGET] + list(GROUP_COLUMNS)].copy()
    test_output["prediction"] = test_predictions
    test_output["absolute_error"] = (test_output[TARGET] - test_output["prediction"]).abs()
    test_output.to_csv(PREDICTIONS_PATH, index=False)
    save_figures(train_df, y_test, test_predictions, importances)

    production_pipeline = model_candidates()[selected_name]
    production_pipeline.fit(x, y)
    joblib.dump(production_pipeline, MODEL_PATH)

    metadata = {
        "project": "Student Performance Predictor",
        "target": TARGET,
        "selected_model": selected_name,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_metrics": test_metrics,
        "cv": cv_details,
        "dataset_manifest": dataset_manifest(DATA_PATH),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "notes": [
            "Les métriques de test sont calculées avant le réentraînement final sur toutes les données.",
            "Les importances décrivent des associations prédictives et non des effets causaux.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    DATASET_MANIFEST_PATH.write_text(
        json.dumps(metadata["dataset_manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    st.subheader(t("Métriques finales sur le jeu de test", "Final test-set metrics"))
    st.json(test_metrics)
    st.success(f"{t('Modèle enregistré :', 'Model saved:')} {MODEL_PATH}")


