import numpy as np
import wfdb

from src.config.constants import MITDB_PATH

from src.datasets.record_processor import process_record


def build_dataset(
    record_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_X = []
    all_y = []
    all_patient_ids = []

    for record_id in record_ids:
        print(f"Processing record {record_id}...")

        record_path = f"{MITDB_PATH}/{record_id}"

        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")

        signal = record.p_signal[:, 0]

        X_record, y_record = process_record(signal, annotation)
        patient_record_ids = np.full(len(y_record),int(record_id))

        all_X.append(X_record)
        all_y.append(y_record)
        all_patient_ids.append(patient_record_ids)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    patient_ids = np.concatenate(all_patient_ids)

    return X, y, patient_ids