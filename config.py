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
