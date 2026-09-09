import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# File Paths
RAW_DATA_FILE = os.path.join(DATA_DIR, "seed_reports (1).csv")
DASHBOARD_OUTPUT_FILE = os.path.join(DATA_DIR, "unsafe_dashboard_historical.csv")
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "unsafe_embeddings.npy")
MODEL_FILE = os.path.join(MODELS_DIR, "xgboost_unsafe_model.pkl")

# ML Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUBMISSIONS_CSV = os.path.join(DATA_DIR, "unsafe_submissions.csv")