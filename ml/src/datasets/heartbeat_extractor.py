import numpy as np
from src.config.constants import (
    SAMPLES_BEFORE_R,
    SAMPLES_AFTER_R,
)


def extract_heartbeat(
    signal: np.ndarray,
    r_peak: int
) -> np.ndarray | None:

    start = r_peak - SAMPLES_BEFORE_R
    end = r_peak + SAMPLES_AFTER_R

    if start < 0 or end > len(signal):
        return None

    heartbeat = signal[start:end]

    return heartbeat.copy()