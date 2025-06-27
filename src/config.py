from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- PATHS ---
DATA_DIR = BASE_DIR / "data"
STORES_PATH = DATA_DIR / "stores.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sampleSubmission.csv"

# --- DATA SPLITTING & FEATURE PARAMS ---

# Note: This simple ratio split is often replaced with a more robust
# time-series validation strategy later in a project.
TRAIN_RATIO = 0.8 
VAL_RATIO = 0.2

TARGET_COLUMN = "Weekly_Sales"
DATE_COLUMN = "Date"