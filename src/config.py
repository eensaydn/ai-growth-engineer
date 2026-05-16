"""Central configuration: paths, hyperparameters, constants."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

PROFILES_CSV = DATA_RAW / "advanced_user_profiles_with_uuid.csv"
EVENTS_CSV = DATA_RAW / "advanced_user_events_with_uuid.csv"

USER_FEATURES_CSV = DATA_PROCESSED / "user_features.csv"

RF_MODEL_PATH = MODELS_DIR / "rf_payment_classifier.joblib"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_segmenter.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

PREDICTIONS_CSV = OUTPUTS_DIR / "user_predictions.csv"
PREDICTIONS_JSON = OUTPUTS_DIR / "user_predictions.json"
SEGMENT_SUMMARY_CSV = OUTPUTS_DIR / "segment_summary.csv"
FEATURE_IMPORTANCES_CSV = OUTPUTS_DIR / "feature_importances.csv"
EVALUATION_REPORT = OUTPUTS_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_PNG = OUTPUTS_DIR / "confusion_matrix.png"
CLUSTER_VIZ_PNG = OUTPUTS_DIR / "cluster_visualization.png"
ELBOW_PLOT_PNG = OUTPUTS_DIR / "elbow_plot.png"

# Random seed
RANDOM_STATE = 42

# Event types in the dataset
EVENT_TYPES = [
    "login", "logout", "checkout_start",
    "payment_success", "trial_extension", "feature_click",
]

# Categorical columns from profiles
PROFILE_CATEGORICAL = ["plan_type", "country", "device_type", "industry"]

# Plan ordinal mapping (Free -> Pro -> Business)
PLAN_ORDINAL = {"Free": 0, "Pro": 1, "Business": 2}

# Plan multiplier for growth potential (Business users have no upsell target)
PLAN_GROWTH_MULTIPLIER = {"Free": 1.0, "Pro": 0.7, "Business": 0.0}

# Reference date for recency = max(event timestamp) + 1 day; resolved at runtime.
# Window length used by activity_trend (last_7 vs first_7).
WINDOW_DAYS = 28
RECENT_WINDOW_DAYS = 7

# Columns that leak the target (n_payment_success) — excluded from RF training.
LEAKAGE_COLS = [
    "n_payment_success",
    "checkout_to_payment_rate",
    "login_to_payment_rate",
    "payment_per_event",
    "recency_payment_days",
    "is_high_payer",
]

# K-Means feature subset (curated 8 behavioral features)
KMEANS_FEATURES = [
    "n_login", "n_checkout_start", "n_payment_success",
    "n_feature_click", "n_trial_extension",
    "activity_trend", "last_event_day", "checkout_to_payment_rate",
]

# Random Forest hyperparameters
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Cross-validation folds
CV_FOLDS = 5

# K-Means
N_CLUSTERS = 4
KMEANS_PARAMS = dict(
    n_clusters=N_CLUSTERS,
    n_init=20,
    max_iter=500,
    random_state=RANDOM_STATE,
)

# Segmentation thresholds (tuned post-hoc to yield a balanced distribution
# given the 28-day data window). See README for the empirical rationale.
CHURN_THRESHOLD = 0.65
GROWTH_THRESHOLD = 0.40
HIGH_VALUE_PROB_THRESHOLD = 0.55

# Target tertile percentile cutoff for is_high_payer
HIGH_PAYER_PERCENTILE = 0.67
