import numpy as np

from src.config.constants import (
    TRAIN_PATIENTS,
    VALIDATION_PATIENTS,
    TEST_PATIENTS,
    SPLITS_PATH
)


def save_patient_splits(
    train_patients: np.ndarray,
    val_patients: np.ndarray,
    test_patients: np.ndarray
) -> None:


    SPLITS_PATH.mkdir(parents=True, exist_ok=True)

    np.save(
        SPLITS_PATH / "train_patients.npy",
        train_patients
    )

    np.save(
        SPLITS_PATH / "val_patients.npy",
        val_patients
    )

    np.save(
        SPLITS_PATH / "test_patients.npy",
        test_patients
    )


def load_patient_splits():


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
        test_patients
    )


def validate_patient_splits(
    train_patients: np.ndarray,
    val_patients: np.ndarray,
    test_patients: np.ndarray
) -> None:

    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    if train_set & val_set:
        raise ValueError(
            "Patient overlap detected between training and validation sets."
        )

    if train_set & test_set:
        raise ValueError(
            "Patient overlap detected between training and test sets."
        )

    if val_set & test_set:
        raise ValueError(
            "Patient overlap detected between validation and test sets."
        )


def get_patient_splits():


    train_patients = np.array(TRAIN_PATIENTS)
    val_patients = np.array(VALIDATION_PATIENTS)
    test_patients = np.array(TEST_PATIENTS)

    validate_patient_splits(
        train_patients,
        val_patients,
        test_patients
    )

    split_files_exist = (
        (SPLITS_PATH / "train_patients.npy").exists()
        and (SPLITS_PATH / "val_patients.npy").exists()
        and (SPLITS_PATH / "test_patients.npy").exists()
    )

    if not split_files_exist:
        save_patient_splits(
            train_patients,
            val_patients,
            test_patients
        )

    return load_patient_splits()