from pathlib import Path

SAMPLES_BEFORE_R = 36
SAMPLES_AFTER_R = 53

HEARTBEAT_LENGTH = (
    SAMPLES_BEFORE_R +
    SAMPLES_AFTER_R
)

TARGET_CLASSES = {
    "N", 
    "S",  
    "V", 
}



PROJECT_ROOT = Path(__file__).resolve().parents[2]
SVDB_PATH = PROJECT_ROOT / "data" / "raw" / "svdb"

SVDB_RECORDS = sorted(
    header.stem
    for header in SVDB_PATH.glob("*.hea")
)


TRAIN_PATIENTS = [
    800, 801, 803, 804, 805, 806, 807, 808, 809, 812, 820, 821, 823, 824, 
    827, 828, 842, 846, 848, 849, 853, 854, 855, 856, 857, 858, 859, 861, 
    862, 863, 864, 865, 866, 868, 870, 871, 872, 873, 874, 875, 877, 878, 
    879, 880, 881, 882, 883, 886, 888, 889, 890, 891, 893, 894
]

VALIDATION_PATIENTS = [
    811, 822, 825, 826, 841, 844, 845, 847, 850, 884, 887, 892
]

TEST_PATIENTS = [
    802, 810, 829, 840, 843, 851, 852, 860, 867, 869, 876, 885
]


PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

SPLITS_PATH = PROCESSED_DATA_PATH / "splits"


LABEL_TO_INDEX = {
    "N": 0,
    "S": 1,
    "V": 2,
}

INDEX_TO_LABEL = {
    0: "N",
    1: "S",
    2: "V",
}

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

RANDOM_SEED = 42

NUM_CLASSES = 3

CLASS_NAMES = [
    "N",
    "S",
    "V",
]

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Saved Models Path
SAVED_MODELS_PATH = PROJECT_ROOT / "saved_models"

BASELINE_MODEL_PATH = (
    SAVED_MODELS_PATH /
    "baseline_svdb_best.pth"
)

REGULARIZED_MODEL_PATH = (
    SAVED_MODELS_PATH / 
    "regularized_svdb_best.pth"
)

DEEPER_MODEL_PATH = (
    SAVED_MODELS_PATH / 
    "deeper_svdb_best.pth"
)

RESNET1D_MODEL_PATH = (
    SAVED_MODELS_PATH / 
    "resnet1d_svdb_best.pth"
)

EARLY_STOPPING_PATIENCE = 10

WEIGHT_DECAY = 1e-4

DROPOUT_RATE = 0.3