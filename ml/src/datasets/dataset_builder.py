import numpy as np
import wfdb

from src.config.constants import MITDB_PATH

from src.datasets.record_processor import process_record


def build_dataset(
    record_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:

    all_X = []
    all_y = []

    for record_id in record_ids:
        print(f"Processing record {record_id}...")

        record_path = f"{MITDB_PATH}/{record_id}"

        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")

        signal = record.p_signal[:, 0]

        X_record, y_record = process_record(signal, annotation)

        all_X.append(X_record)
        all_y.append(y_record)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    return X, y