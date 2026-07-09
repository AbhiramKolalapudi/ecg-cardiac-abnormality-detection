import numpy as np


def extract_heartbeat(
    signal: np.ndarray,
    r_peak: int,
    samples_before: int = 100,
    samples_after: int = 150
) -> np.ndarray | None:

    start = r_peak - samples_before
    end = r_peak + samples_after

    if start < 0 or end > len(signal):
        return None

    heartbeat = signal[start:end]

    return heartbeat.copy()