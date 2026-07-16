import torch

from src.training.metrics import compute_metrics


def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    val_loss = running_loss / len(val_loader)

    metrics = compute_metrics(
        all_labels,
        all_predictions
    )

    metrics["loss"] = val_loss

    return metrics