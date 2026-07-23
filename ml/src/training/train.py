import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.config.constants import (
    MITBIH_RECORDS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    RESNET1D_MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    WEIGHT_DECAY
)

from src.datasets.dataset_builder import build_dataset
from src.datasets.ecg_dataset import ECGDataset

from src.preprocessing.pipeline import prepare_datasets

from src.models.resnet1d import ResNet1D
from src.training.trainer import train_one_epoch
from src.training.validator import validate_one_epoch


def main():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    X, y, patient_ids = build_dataset(
        MITBIH_RECORDS
    )

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

    model = ResNet1D().to(device)

    print("\nModel Architecture:")
    print(model)

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

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    best_f1 = -float("inf")
    best_epoch = -1
    patience_counter = 0

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

        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        current_f1 = val_metrics["f1"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1
            best_val_metrics = val_metrics

            torch.save(model.state_dict(),RESNET1D_MODEL_PATH,)

            patience_counter = 0

            print(f"\nNew best model saved (Macro F1 = {best_f1:.4f})")
        
        else:
            patience_counter += 1


        print(f"Train Loss: {train_loss:.4f}")

        print(f"Val Loss: {val_metrics['loss']:.4f}")

        print(f"Accuracy: {val_metrics['accuracy']:.4f}")

        print(f"Precision: {val_metrics['precision']:.4f}")

        print(f"Recall: {val_metrics['recall']:.4f}")

        print(f"F1 Score: {val_metrics['f1']:.4f}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping triggered.")
            break

    print("\nTraining Complete!")
    CLASS_NAMES = ["N", "A", "V", "L", "R"]

    print("\nPer-Class Metrics")
    print("=" * 50)

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"\nClass {class_name}")
        print(f"Precision: {best_val_metrics['per_class_precision'][i]:.4f}")
        print(f"Recall:    {best_val_metrics['per_class_recall'][i]:.4f}")
        print(f"F1 Score:  {best_val_metrics['per_class_f1'][i]:.4f}")

    print("\nConfusion Matrix")
    print("=" * 50)
    print(best_val_metrics["confusion_matrix"])

    print("\nBest Model")
    print("=" * 50)
    print(f"Epoch: {best_epoch}")
    print(f"Macro F1: {best_f1:.4f}")
    print(f"Saved to: {RESNET1D_MODEL_PATH}")


if __name__ == "__main__":
    main()