import numpy as np

from src.config.constants import (
    LABEL_TO_INDEX,
    INDEX_TO_LABEL
)


def encode_labels(
    labels: np.ndarray
) -> np.ndarray:

    encoded_labels = np.array(
        [
            LABEL_TO_INDEX[label]
            for label in labels
        ]
    )

    return encoded_labels


def decode_labels(
    labels: np.ndarray
) -> np.ndarray:

    decoded_labels = np.array(
        [
            INDEX_TO_LABEL[label]
            for label in labels
        ]
    )

    return decoded_labels