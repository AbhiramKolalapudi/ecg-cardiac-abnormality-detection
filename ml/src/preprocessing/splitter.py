import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

from src.config.constants import (
    SPLITS_PATH,
    RANDOM_STATE,
)

def save_patient_splits(
    train_patients: np.ndarray,
    val_patients: np.ndarray,
    test_patients: np.ndarray,
) -> None:

    SPLITS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        SPLITS_PATH / "train_patients.npy",
        train_patients,
    )

    np.save(
        SPLITS_PATH / "val_patients.npy",
        val_patients,
    )

    np.save(
        SPLITS_PATH / "test_patients.npy",
        test_patients,
    )

def load_patient_splits(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    train_patients = np.load(
        SPLITS_PATH / "train_patients.npy"
    )

    val_patients = np.load(
        SPLITS_PATH / "val_patients.npy"
    )

    test_patients = np.load(
        SPLITS_PATH / "test_patients.npy"
    )

    return (
        train_patients,
        val_patients,
        test_patients,
    )

def generate_patient_splits(
    y: np.ndarray,
    patient_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    splitter = StratifiedGroupKFold(
        n_splits=7,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(
        splitter.split(
            X=np.zeros(len(y)),
            y=y,
            groups=patient_ids,
        )
    )

    test_indices = folds[0][1]
    val_indices = folds[1][1]

    train_indices = np.concatenate(
        [fold[1] for fold in folds[2:]]
    )

    test_patients = np.unique(
        patient_ids[test_indices]
    )

    val_patients = np.unique(
        patient_ids[val_indices]
    )

    train_patients = np.unique(
        patient_ids[train_indices]
    )

    return (
        train_patients,
        val_patients,
        test_patients,
    )

def get_patient_splits(
    y: np.ndarray,
    patient_ids: np.ndarray,
    regenerate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if not regenerate:
        try:
            return load_patient_splits()

        except FileNotFoundError:
            pass

    train_patients, val_patients, test_patients = (
        generate_patient_splits(
            y,
            patient_ids,
        )
    )

    save_patient_splits(
        train_patients,
        val_patients,
        test_patients,
    )

    return (
        train_patients,
        val_patients,
        test_patients,
    )