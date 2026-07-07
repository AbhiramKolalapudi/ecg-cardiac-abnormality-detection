import wfdb
import numpy as np
import matplotlib.pyplot as plt

record_path = "ml/data/raw/mit-bih-arrhythmia-database/100"

record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, "atr")

lead1 = record.p_signal[:1000, 0]
time = np.arange(len(lead1)) / record.fs

mask = annotation.sample < 1000

annotation_samples = annotation.sample[mask]
annotation_symbols = np.array(annotation.symbol)[mask]

annotation_time = annotation_samples / record.fs
annotation_voltage = lead1[annotation_samples]

plt.figure(figsize=(12, 4))

plt.plot(time, lead1, label="Lead MLII")

plt.scatter(annotation_time, annotation_voltage, color="red", s=30)

for x, y, symbol in zip(annotation_time, annotation_voltage, annotation_symbols):
    plt.text(x, y + 0.08, symbol, fontsize=8, ha="center")

plt.title("ECG Record 100 with Annotations")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.legend()

plt.show()

before = 100
after = 150

# Using a valid R peak for demonstration
r_peak = 370

heartbeat = lead1[r_peak-before:r_peak+after]

print("\nExtracted Heartbeat Shape:", heartbeat.shape)

plt.figure(figsize=(8, 4))

plt.plot(heartbeat)

plt.title("Extracted Heartbeat")
plt.xlabel("Sample")
plt.ylabel("Voltage (mV)")
plt.grid(True)

plt.show()