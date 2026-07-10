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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MITDB_PATH = PROJECT_ROOT / "data" / "raw" / "mitdb"