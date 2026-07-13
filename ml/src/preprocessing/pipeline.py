import numpy as np

from src.preprocessing.splitter import get_patient_splits
from src.preprocessing.normalizer import normalize_dataset
from src.preprocessing.encoder import encode_labels


def prepare_datasets(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
):

    train_patients, val_patients, test_patients = (
        get_patient_splits()
    )

    train_mask = np.isin(
        patient_ids,
        train_patients
    )

    val_mask = np.isin(
        patient_ids,
        val_patients
    )

    test_mask = np.isin(
        patient_ids,
        test_patients
    )

    X_train = X[train_mask]
    y_train = y[train_mask]
    train_patient_ids = patient_ids[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]
    val_patient_ids = patient_ids[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    test_patient_ids = patient_ids[test_mask]

    X_train = normalize_dataset(X_train)
    X_val = normalize_dataset(X_val)
    X_test = normalize_dataset(X_test)

    y_train = encode_labels(y_train)
    y_val = encode_labels(y_val)
    y_test = encode_labels(y_test)

    return (
        X_train,
        y_train,
        train_patient_ids,
        X_val,
        y_val,
        val_patient_ids,
        X_test,
        y_test,
        test_patient_ids,
    )