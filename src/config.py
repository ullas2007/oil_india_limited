from pathlib import Path


# config.py is inside src/
# src/config.py -> src/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent


# Main project folders
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
UPLOADS_DIR = DATA_DIR / "uploads"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

MODELS_DIR = BASE_DIR / "models"


# Upload folders
IMAGE_UPLOAD_DIR = UPLOADS_DIR / "images"
AUDIO_UPLOAD_DIR = UPLOADS_DIR / "audio"


# Data files
RAW_DATA_FILE = RAW_DIR / "near_miss_800_varied_dataset.csv"

STANDARDIZED_FILE = PROCESSED_DIR / "near_miss_standardized.csv"
ML_READY_FILE = PROCESSED_DIR / "near_miss_ml_ready.csv"
EMBEDDINGS_FILE = PROCESSED_DIR / "near_miss_embeddings.npy"

DASHBOARD_OUTPUT_FILE = OUTPUTS_DIR / "near_miss_dashboard_output.csv"
SUBMISSIONS_CSV = SUBMISSIONS_DIR / "employee_submissions.csv"


# Model files
SIF_MODEL_FILE = MODELS_DIR / "xgboost_sif_model.pkl"
RISK_MODEL_FILE = MODELS_DIR / "xgboost_risk_model.pkl"
RISK_LABEL_ENCODER_FILE = MODELS_DIR / "risk_label_encoder.pkl"


# NLP model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Risk thresholds
SIF_THRESHOLD = 0.35
MEDIUM_THRESHOLD = 0.30


def create_project_directories():
    directories = [
        RAW_DIR,
        PROCESSED_DIR,
        OUTPUTS_DIR,
        UPLOADS_DIR,
        IMAGE_UPLOAD_DIR,
        AUDIO_UPLOAD_DIR,
        SUBMISSIONS_DIR,
        MODELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)