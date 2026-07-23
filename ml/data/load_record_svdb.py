from collections import Counter
import numpy as np

from src.config.constants import SVDB_RECORDS
from src.datasets.dataset_builder import build_dataset
from src.preprocessing.pipeline import prepare_datasets
from src.preprocessing.encoder import decode_labels


def print_distribution(name, labels):
    if np.issubdtype(labels.dtype, np.integer):
        labels = decode_labels(labels)
    counts = Counter(labels)

    total = len(labels)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Total Heartbeats: {total}")

    for cls in ["N", "S", "V"]:
        count = counts[cls]
        percentage = 100 * count / total
        print(f"{cls}: {count:6d} ({percentage:.2f}%)")


print("=" * 60)
print("BUILDING DATASET")
print("=" * 60)

X, y, patient_ids = build_dataset(SVDB_RECORDS)

print("\nOverall Dataset")
print("-" * 40)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("patient_ids shape:", patient_ids.shape)

print("\nHeartbeat Length:", X.shape[1])
print("Number of Records:", len(np.unique(patient_ids)))

print_distribution("Overall Distribution", y)

(
    X_train,
    y_train,
    train_patient_ids,
    X_val,
    y_val,
    val_patient_ids,
    X_test,
    y_test,
    test_patient_ids,
) = prepare_datasets(
    X,
    y,
    patient_ids,
)

print("\n" + "=" * 60)
print("TRAIN / VALIDATION / TEST")
print("=" * 60)

print("\nTrain")
print("X:", X_train.shape)
print("y:", y_train.shape)
print("Patients:", len(np.unique(train_patient_ids)))
print_distribution("Train Distribution", y_train)

print("\nValidation")
print("X:", X_val.shape)
print("y:", y_val.shape)
print("Patients:", len(np.unique(val_patient_ids)))
print_distribution("Validation Distribution", y_val)

print("\nTest")
print("X:", X_test.shape)
print("y:", y_test.shape)
print("Patients:", len(np.unique(test_patient_ids)))
print_distribution("Test Distribution", y_test)

print("\n" + "=" * 60)
print("NORMALIZATION CHECK")
print("=" * 60)

print("Train Mean:", X_train.mean())
print("Train Std :", X_train.std())

print("Validation Mean:", X_val.mean())
print("Validation Std :", X_val.std())

print("Test Mean:", X_test.mean())
print("Test Std :", X_test.std())