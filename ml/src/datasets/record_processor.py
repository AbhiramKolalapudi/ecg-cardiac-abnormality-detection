import numpy as np
from wfdb import Annotation
from src.config.constants import TARGET_CLASSES

from .heartbeat_extractor import extract_heartbeat


def process_record(
    signal: np.ndarray,
    annotation: Annotation
) -> tuple[np.ndarray, np.ndarray]:

    heartbeats = []
    labels = []

    for r_peak, symbol in zip(annotation.sample, annotation.symbol):

        if symbol not in TARGET_CLASSES:
            continue

        heartbeat = extract_heartbeat(
            signal=signal,
            r_peak=r_peak
        )

        if heartbeat is None:
            continue

        heartbeats.append(heartbeat)
        labels.append(symbol)

    X_record = np.array(heartbeats)
    y_record = np.array(labels)

    return X_record, y_record