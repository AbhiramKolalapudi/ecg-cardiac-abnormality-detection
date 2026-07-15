import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.config.constants import (
    MITBIH_RECORDS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
)

from src.datasets.dataset_builder import build_dataset
from src.datasets.ecg_dataset import ECGDataset

from src.preprocessing.pipeline import prepare_datasets

from src.models.baseline_cnn import BaselineCNN
from src.training.trainer import train_one_epoch


def main():

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Build complete dataset
    X, y, patient_ids = build_dataset(
        MITBIH_RECORDS
    )

    # Train / Validation / Test split
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

    # PyTorch datasets
    train_dataset = ECGDataset(
        X_train,
        y_train,
    )

    val_dataset = ECGDataset(
        X_val,
        y_val,
    )

    test_dataset = ECGDataset(
        X_test,
        y_test,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Model
    model = BaselineCNN().to(device)

    print("\nModel Architecture:")
    print(model)

    # Class weights for imbalance handling
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    ).to(device)

    print("\nClass Weights:")
    print(class_weights)

    # Loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Training loop
    for epoch in range(NUM_EPOCHS):

        print(
            f"\n{'=' * 50}"
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
        )

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"Training Loss: "
            f"{train_loss:.4f}"
        )

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()