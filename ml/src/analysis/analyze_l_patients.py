import numpy as np

from src.config.constants import MITBIH_RECORDS
from src.datasets.dataset_builder import build_dataset

X, y, patient_ids = build_dataset(MITBIH_RECORDS)

L_CLASS = 3

print("Patients containing L beats")
print("=" * 40)

patients = np.unique(patient_ids[y == "L"])

for patient in patients:
    count = np.sum((patient_ids == patient) & (y == "L"))
    print(f"Patient {patient}: {count}")