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
TRAIN_PATIENTS = [
    100, 101, 103, 105, 106, 108, 109, 112, 114, 115,
    116, 118, 121, 122, 123, 200, 203, 205, 208, 209,
    210, 212, 213, 214, 215, 219, 221, 230, 232, 233,
    234
]

VALIDATION_PATIENTS = [
    117, 119, 201, 202, 207, 223, 228, 231
]

TEST_PATIENTS = [
    102, 104, 107, 111, 113, 124, 217, 220, 222
]


# Processed Data Paths
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

SPLITS_PATH = PROCESSED_DATA_PATH / "splits"


#Label Mapping
LABEL_TO_INDEX = {
    "N": 0,
    "A": 1,
    "V": 2,
    "L": 3,
    "R": 4,
}

INDEX_TO_LABEL = {
    0: "N",
    1: "A",
    2: "V",
    3: "L",
    4: "R",
}

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

# Reproducibility
RANDOM_SEED = 42

# Dataset Information
NUM_CLASSES = 5

# Ordered Class Names
CLASS_NAMES = [
    "N",
    "A",
    "V",
    "L",
    "R",
]

import torch

# Preferred Training Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Saved Models Path
SAVED_MODELS_PATH = PROJECT_ROOT / "saved_models"

BASELINE_MODEL_PATH = (
    SAVED_MODELS_PATH /
    "baseline_best.pth"
)

REGULARIZED_MODEL_PATH = (
    SAVED_MODELS_PATH / 
    "regularized_best.pth"
)

DEEPER_MODEL_PATH = (
    SAVED_MODELS_PATH / 
    "deeper_best.pth"
)

RESNET1D_MODEL_PATH = (
    SAVED_MODELS_PATH / 
    "resnet1d_best.pth"
)

EARLY_STOPPING_PATIENCE = 5

WEIGHT_DECAY = 1e-4

DROPOUT_RATE = 0.3