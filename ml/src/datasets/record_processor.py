import numpy as np
from wfdb import Annotation

from .heartbeat_extractor import extract_heartbeat


def process_record(
    signal: np.ndarray,
    annotation: Annotation,
    valid_labels: set[str],
    samples_before: int = 100,
    samples_after: int = 150,
) -> tuple[np.ndarray, np.ndarray]:

    heartbeats = []
    labels = []

    for r_peak, symbol in zip(annotation.sample, annotation.symbol):

        if symbol not in valid_labels:
            continue

        heartbeat = extract_heartbeat(
            signal=signal,
            r_peak=r_peak,
            samples_before=samples_before,
            samples_after=samples_after,
        )

        if heartbeat is None:
            continue

        heartbeats.append(heartbeat)
        labels.append(symbol)

    X_record = np.array(heartbeats)
    y_record = np.array(labels)

    return X_record, y_record