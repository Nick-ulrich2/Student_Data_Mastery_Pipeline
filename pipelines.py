from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from train import *
from config import *
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from translations import t
from themes import configure_matplotlib_theme, theme_chart_colors

import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor() -> ColumnTransformer:
    # Gestion des valeurs manquantes et normalisation des variables numeriques
    numeric_pipeline = Pipeline(
        steps=[
            # Remplacons les valeurs manquantes par la mediane
            ("imputer", SimpleImputer(strategy="median")),
            # Normalisons ces valeurs 
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            # Remplacons les valeurs manquantes par les valeurs regulieres
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Normalisation et transformation des variables nominales
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

# Cette fonction construit une pipeline (Processus) constituee de 2 etapes : 
    # Preparation des donnees 
    # Modele de prediction
    '''
        Sans cette pipeline, il faudrait 
        préparer les données manuellement avant chaque entraînement et chaque prédiction. 
        Cela pourrait provoquer des erreurs ou des différences entre l’entraînement et l’utilisation réelle.
    '''
def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", model),
        ]
    )

# Cette fonction retourne un dictionnaire contenant differents models de machine learning en cles et des pipelines en valeurs
def model_candidates() -> dict[str, Pipeline]:
    return {
        "dummy_mean": make_pipeline(DummyRegressor(strategy="mean")), # Predit toujours la moyenne des notes 
        "ridge": make_pipeline(Ridge(alpha=10.0)), # Cherche la relation entre les entrees etr la target 
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
    
# Cette fonction mesure la qualite de la prediction
def metric_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Ces metriques repondent respectivement a chacune de ces questions:
    return {
        # MAE -> En moyenne, je me trompe de combien de points? Il donne la moyenne des erreurs entre les vraies notes et les predictions 
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # RMSE -> Quelle est l'erreur moyenne ? (Ceci est exprimee dans l'unite de la note)
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        # R2 -> Quelle proportion de la variation des notes mon modele arrive t'il a exprimer?
        "r2": float(r2_score(y_true, y_pred)),
    }


# def dataset_manifest(path: Path) -> dict[str, object]:
#     digest = hashlib.sha256(path.read_bytes()).hexdigest()
#     return {
#         "file_name": path.name,
#         "sha256": digest,
#         "bytes": path.stat().st_size,
#         "rows": int(pd.read_csv(path).shape[0]),
#         "columns": int(pd.read_csv(path).shape[1]),
#     }


def cross_validate_candidates(candidates: dict[str, Pipeline], x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []
    cv_details: dict[str, dict[str, float]] = {}
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}

    for name, pipeline in candidates.items():
        result = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
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
        # Ici je dois ecrire un code streamlit, qui affiche chaque resultat pour chaque model
        
        rows.append(row)
        cv_details[name] = {key: value for key, value in row.items() if key != "model"}

    comparison = pd.DataFrame(rows).sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)
    return comparison, cv_details


def aggregate_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    if hasattr(model, "feature_importances_"): # Si le model teste a un rapport avec forets d'arbres
        values = np.asarray(model.feature_importances_, dtype=float)
        direction = "importance"
    elif hasattr(model, "coef_"): # Si ca concerne les modeles lineaires comme ridge
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        direction = "absolute_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "source", "kind"])

    rows = []
    all_source_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for name, value in zip(feature_names, values):
        source = name
        for column in all_source_columns:
            if name == f"num__{column}" or name.startswith(f"cat__{column}_"):
                source = column
                break
        rows.append({"feature": name, "importance": float(value), "source": source, "kind": direction})
    result = pd.DataFrame(rows)
    # Inserer du code afin que ces colonnes soient classees du plus important au moins important a l'aide d'un graphique et affiche avec streamlit
    return result.groupby(["source", "kind"], as_index=False)["importance"].sum().sort_values("importance", ascending=False)

# Cette fonction mesure la qualité du modèle pour différents sous-groupes.
def subgroup_metrics(test_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
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


def save_figures(train_df: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray, importances: pd.DataFrame) -> None:
    # Ajouter un code streamlit pour dire ou afficher chacun des graphiques
    configure_matplotlib_theme()
    chart_colors = theme_chart_colors()
    plt.figure(figsize=(9, 5))
    sns.histplot(train_df[TARGET], bins=30, kde=True, color=chart_colors["blue"])
    plt.title(t("Distribution de Exam_Score", "Exam_Score distribution"))
    plt.xlabel(t("Exam_Score", "Exam_Score"))
    plt.tight_layout()
    # plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=160)
    plt.close()

    residuals = y_test.to_numpy() - predictions
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.55, color=chart_colors["orange"])
    plt.axhline(0, color=chart_colors["text"], linewidth=1)
    plt.title(t("Résidus sur le jeu de test", "Residuals on the test set"))
    plt.xlabel(t("Prédiction", "Prediction"))
    plt.ylabel(t("Résidu réel - prédit", "Actual minus predicted residual"))
    plt.tight_layout()
    # plt.savefig(FIGURES_DIR / "residuals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.55, color=chart_colors["green"])
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(limits, limits, linestyle="--", color=chart_colors["text"])
    plt.title(t("Scores réels et prédits", "Actual and predicted scores"))
    plt.xlabel(t("Score réel", "Actual score"))
    plt.ylabel(t("Score prédit", "Predicted score"))
    plt.tight_layout()
    # plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    if not importances.empty:
        # top = importances.head(12).sort_values("importance")
        plt.figure(figsize=(9, 6))
        sns.barplot(data=importances, x="importance", y="source", color=chart_colors["blue"])
        plt.title(t("Variables les plus importantes pour le modèle retenu", "Most important variables for the selected model"))
        plt.xlabel(t("Importance agrégée", "Aggregated importance"))
        plt.ylabel(t("Variable source", "Source variable"))
        plt.tight_layout()
        # plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
        plt.close()


# def main() -> None:
#     # ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
#     # MODELS_DIR.mkdir(parents=True, exist_ok=True)

#     df = load_data(DATA_PATH)
#     validate_training_frame(df)
#     x = feature_frame(df)
#     y = target_series(df)
#     x_train, x_test, y_train, y_test, _, test_indices = train_test_split(
#         x,
#         y,
#         df.index,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#     )
#     train_df = df.loc[x_train.index].copy()
#     test_df = df.loc[x_test.index].copy()

#     candidates = model_candidates()
#     comparison, cv_details = cross_validate_candidates(candidates, x_train, y_train)
#     selected_name = str(comparison.iloc[0]["model"])
#     selected_pipeline = candidates[selected_name]
#     selected_pipeline.fit(x_train, y_train)
#     test_predictions = selected_pipeline.predict(x_test)
#     test_metrics = metric_dict(y_test, test_predictions)

#     importances = aggregate_importance(selected_pipeline)
#     importances.to_csv(IMPORTANCE_PATH, index=False)
#     subgroup = subgroup_metrics(test_df, test_predictions)
#     subgroup.to_csv(SUBGROUP_PATH, index=False)
#     comparison.to_csv(COMPARISON_PATH, index=False)

#     test_output = test_df[[TARGET] + GROUP_COLUMNS].copy()
#     test_output["prediction"] = test_predictions
#     test_output["absolute_error"] = (test_output[TARGET] - test_output["prediction"]).abs()
#     test_output.to_csv(PREDICTIONS_PATH, index=False)
#     save_figures(train_df, y_test, test_predictions, importances)

#     # Le pipeline de production est ensuite ajusté sur toutes les données disponibles.
#     production_pipeline = model_candidates()[selected_name]
#     production_pipeline.fit(x, y)
#     joblib.dump(production_pipeline, MODEL_PATH)

#     metadata = {
#         "project": "Student Performance Predictor",
#         "target": TARGET,
#         "selected_model": selected_name,
#         "features": FEATURES,
#         "numeric_features": NUMERIC_FEATURES,
#         "categorical_features": CATEGORICAL_FEATURES,
#         "random_state": RANDOM_STATE,
#         "test_size": TEST_SIZE,
#         "cv_folds": CV_FOLDS,
#         "training_rows": int(len(x_train)),
#         "test_rows": int(len(x_test)),
#         "test_metrics": test_metrics,
#         "cv": cv_details,
#         "dataset_manifest": dataset_manifest(DATA_PATH),
#         "python": platform.python_version(),
#         "pandas": pd.__version__,
#         "scikit_learn": __import__("sklearn").__version__,
#         "notes": [
#             "Les métriques de test ont été calculées avant le réentraînement de production sur toutes les lignes.",
#             "Les scores Exam_Score supérieurs à 100 sont conservés car observés dans le fichier source.",
#             "Les importances décrivent des associations prédictives et non des effets causaux.",
#         ],
#     }
#     METRICS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
#     DATASET_MANIFEST_PATH.write_text(json.dumps(metadata["dataset_manifest"], indent=2, ensure_ascii=False), encoding="utf-8")

#     print(json.dumps({"selected_model": selected_name, "test_metrics": test_metrics, "model_path": str(MODEL_PATH)}, indent=2, ensure_ascii=False))
