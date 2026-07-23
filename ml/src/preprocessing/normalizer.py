import numpy as np


EPSILON = 1e-8


def normalize_heartbeat(
    heartbeat: np.ndarray
) -> np.ndarray:
    mean = heartbeat.mean()
    std = heartbeat.std()

    normalized_heartbeat = (heartbeat - mean) / (std + EPSILON)

    return normalized_heartbeat

def normalize_dataset(
    X: np.ndarray
) -> np.ndarray:
    means = X.mean(
        axis=1,
        keepdims=True
    )

    stds = X.std(
        axis=1,
        keepdims=True
    )

    X_normalized = (X - means) / (stds + EPSILON)

    return X_normalized