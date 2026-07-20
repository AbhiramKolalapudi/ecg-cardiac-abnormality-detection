import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.config.constants import (
    CLASS_NAMES,
    DEVICE,
    MITBIH_RECORDS,
    REGULARIZED_MODEL_PATH,
)
from src.datasets.dataset_builder import build_dataset
from src.models.regularized_cnn import RegularizedCNN
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
    _,
    X_test,
    y_test,
    _
) = prepare_datasets(
    X,
    y,
    patient_ids
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
).unsqueeze(1)

print(f"Validation Tensor Shape: {X_val.shape}")


# ============================================================
# Load Trained Model
# ============================================================

model = RegularizedCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        REGULARIZED_MODEL_PATH,
        map_location=DEVICE,
    )
)

model.eval()

X_val = X_val.to(DEVICE)


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
print("FC2 WEIGHT SIMILARITY")
print("=" * 60)

weights = model.fc2.weight.detach().cpu().numpy()

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