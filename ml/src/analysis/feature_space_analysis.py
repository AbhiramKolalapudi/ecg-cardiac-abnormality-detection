import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

from src.config.constants import (
    CLASS_NAMES,
    DEVICE,
    MITBIH_RECORDS,
    RESNET1D_MODEL_PATH,
)
from src.datasets.dataset_builder import build_dataset
from src.models.resnet1d import ResNet1D
from src.preprocessing.pipeline import prepare_datasets


# ============================================================
# Load Validation Dataset
# ============================================================

X, y, patient_ids = build_dataset(MITBIH_RECORDS)

(
    X_train,
    y_train,
    _,
    X_val,
    y_val,
    patient_ids_val,
    X_test,
    y_test,
    _
) = prepare_datasets(
    X,
    y,
    patient_ids
)

# ============================================================
# Dataset Statistics
# ============================================================

print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Training L Beats   : {np.sum(y_train == 3)}")
print(f"Validation L Beats : {np.sum(y_val == 3)}")
print(f"Test L Beats       : {np.sum(y_test == 3)}")

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
).unsqueeze(1)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
).unsqueeze(1)

print(f"Validation Tensor Shape: {X_val.shape}")


# ============================================================
# Load Trained Model
# ============================================================

model = ResNet1D().to(DEVICE)

model.load_state_dict(
    torch.load(
        RESNET1D_MODEL_PATH,
        map_location=DEVICE,
    )
)

model.eval()

X_val = X_val.to(DEVICE)
X_train = X_train.to(DEVICE)


# ============================================================
# Feature Extraction
# ============================================================

with torch.no_grad():
    features = model.extract_features(X_val)

features = features.cpu().numpy()

print(f"Feature Shape: {features.shape}")


# ============================================================
# t-SNE Visualization
# ============================================================

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
)

features_2d = tsne.fit_transform(features)

print(f"t-SNE Output Shape: {features_2d.shape}")

plt.figure(figsize=(10, 8))

for class_id in range(len(CLASS_NAMES)):

    mask = (y_val == class_id)

    plt.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        s=8,
        alpha=0.6,
        label=CLASS_NAMES[class_id],
    )

plt.title("t-SNE Feature Space")
plt.legend()
plt.show()


# ============================================================
# Logit Analysis
# ============================================================

with torch.no_grad():
    logits = model(X_val)

predictions = torch.argmax(logits, dim=1)

logits = logits.cpu().numpy()
predictions = predictions.cpu().numpy()

print(f"Logit Shape: {logits.shape}")
print(f"Prediction Shape: {predictions.shape}")

# ============================================================
# Classification Report
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_val,
        predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


print("\n" + "=" * 60)
print("TRAINING SET EVALUATION")
print("=" * 60)

with torch.no_grad():
    train_logits = model(X_train)

train_predictions = torch.argmax(train_logits, dim=1).cpu().numpy()

print(
    classification_report(
        y_train,
        train_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


print("\n" + "=" * 60)
print("TRAINING L CLASS ANALYSIS")
print("=" * 60)

train_l_predictions = train_predictions[y_train == 3]

for class_id, class_name in enumerate(CLASS_NAMES):

    count = np.sum(train_l_predictions == class_id)
    percentage = 100 * count / len(train_l_predictions)

    print(
        f"L → {class_name}: {count:4d} ({percentage:.2f}%)"
    )


# ============================================================
# L Class Confusion Analysis
# ============================================================

print("\n" + "=" * 60)
print("L CLASS CONFUSION ANALYSIS")
print("=" * 60)

L_CLASS = 3

l_predictions = predictions[y_val == L_CLASS]

for class_id, class_name in enumerate(CLASS_NAMES):

    count = np.sum(l_predictions == class_id)
    percentage = 100 * count / len(l_predictions)

    print(
        f"L → {class_name}: {count:4d} ({percentage:.2f}%)"
    )


# ============================================================
# Misclassified L Beats
# ============================================================

print("\n" + "=" * 60)
print("MISCLASSIFIED L BEATS")
print("=" * 60)

misclassified = np.where(
    (y_val == L_CLASS) &
    (predictions != L_CLASS)
)[0]

print(f"Total Misclassified L Beats: {len(misclassified)}")


# ============================================================
# Misclassified L Beats by Patient
# ============================================================

print("\n" + "=" * 60)
print("MISCLASSIFIED L BEATS BY PATIENT")
print("=" * 60)

misclassified_patients = patient_ids_val[misclassified]

unique_patients, counts = np.unique(
    misclassified_patients,
    return_counts=True
)

for patient, count in zip(unique_patients, counts):

    print(f"Patient {patient}: {count}")


# ============================================================
# Analyze True L Beats
# ============================================================

L_CLASS = 3

l_indices = np.where(y_val == L_CLASS)[0]

print(f"\nNumber of True L Beats: {len(l_indices)}")


for idx in l_indices[:5]:

    print("=" * 50)
    print(f"Sample Index : {idx}")
    print(f"True Label   : L")
    print(f"Prediction   : {CLASS_NAMES[predictions[idx]]}")

    print("\nLogits:")

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {logits[idx][i]:8.3f}")

print("\n" + "=" * 60)
print("CLASS CENTROID DISTANCES")
print("=" * 60)

centroids = {}

for class_id, class_name in enumerate(CLASS_NAMES):

    centroids[class_name] = features[y_val == class_id].mean(axis=0)

from itertools import combinations

print()

for class1, class2 in combinations(CLASS_NAMES, 2):

    distance = np.linalg.norm(
        centroids[class1] -
        centroids[class2]
    )

    print(
        f"{class1} ↔ {class2}: {distance:.3f}"
    )

print("\n" + "=" * 60)
print("DISTANCE ANALYSIS FOR TRUE L BEATS")
print("=" * 60)

L_FEATURES = features[y_val == 3]

L_centroid = centroids["L"]
V_centroid = centroids["V"]

distance_to_L = np.linalg.norm(
    L_FEATURES - L_centroid,
    axis=1
)

distance_to_V = np.linalg.norm(
    L_FEATURES - V_centroid,
    axis=1
)

closer_to_L = np.sum(distance_to_L < distance_to_V)
closer_to_V = np.sum(distance_to_V < distance_to_L)

print(f"Closer to L centroid : {closer_to_L}")
print(f"Closer to V centroid : {closer_to_V}")

print()

print(f"Average distance to L centroid : {distance_to_L.mean():.3f}")
print(f"Average distance to V centroid : {distance_to_V.mean():.3f}")

print("\n" + "=" * 60)
print("FC WEIGHT SIMILARITY")
print("=" * 60)

weights = model.fc.weight.detach().cpu().numpy()

from itertools import combinations

for i, j in combinations(range(len(CLASS_NAMES)), 2):

    w1 = weights[i]
    w2 = weights[j]

    similarity = (
        np.dot(w1, w2)
        /
        (
            np.linalg.norm(w1)
            * np.linalg.norm(w2)
        )
    )

    print(
        f"{CLASS_NAMES[i]} ↔ {CLASS_NAMES[j]} : {similarity:.3f}"
    )