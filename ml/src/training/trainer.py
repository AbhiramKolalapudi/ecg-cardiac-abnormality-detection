def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0

    for batch_idx, (signals, labels) in enumerate(
        train_loader,
        start=1,
    ):
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(signals)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    return epoch_loss