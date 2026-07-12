from pathlib import Path
# Heartbeat extraction settings
SAMPLES_BEFORE_R = 100
SAMPLES_AFTER_R = 150

HEARTBEAT_LENGTH = (
    SAMPLES_BEFORE_R +
    SAMPLES_AFTER_R
)

# Target heartbeat classes
TARGET_CLASSES = {
    "N",  # Normal beat
    "V",  # Premature ventricular contraction
    "A",  # Atrial premature beat
    "L",  # Left bundle branch block beat
    "R",  # Right bundle branch block beat
}

# Small subset for development and debugging
TEST_RECORDS = [
    "100",
    "101",
    "102",
    "103",
    "104",
]

MITBIH_RECORDS = [
    "100", "101", "102", "103", "104",
    "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115",
    "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201",
    "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214",
    "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231",
    "232", "233", "234"
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MITDB_PATH = PROJECT_ROOT / "data" / "raw" / "mitdb"


# Dataset Split Configuration
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


# Processed Data Paths
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

SPLITS_PATH = PROCESSED_DATA_PATH / "splits"