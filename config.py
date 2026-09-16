from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "SPP.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_HASH_PATH = MODELS_DIR / "model.sha256"
METRICS_PATH = ARTIFACTS_DIR / "metadata.json"
DATASET_MANIFEST_PATH = ARTIFACTS_DIR / "dataset_manifest.json"
FIGURES_DIR = PROJECT_ROOT / "Graph"
LOGO_PATH = PROJECT_ROOT / "Graph" / "logo_spp.webp"
IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
SUBGROUP_PATH = ARTIFACTS_DIR / "subgroup_metrics.csv"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"

TARGET = "Exam_Score"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Class_Size",
]

CATEGORICAL_FEATURES = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Internet_Access",
    "Family_Income",
    "School_Type",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender",
    "Electricity_Access",
    "Region",
    "Transport_Mode",
    "Language_Section",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = FEATURES + [TARGET]
GROUP_COLUMNS = ["Gender", "Region", "School_Type", "Language_Section"]
