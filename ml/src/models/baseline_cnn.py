import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature Extraction Layers
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=5
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Classification Layers
        # 64 channels × 59 samples after Conv/Pool blocks
        self.fc1 = nn.Linear(
            in_features=3776,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=5
        )

    def forward(self, x):

        # Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Classification Head
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        # Return raw logits
        return x