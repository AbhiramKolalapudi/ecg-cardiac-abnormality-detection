import torch.nn as nn

from src.config.constants import NUM_CLASSES
from src.config.constants import DROPOUT_RATE


class RegularizedCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature Extraction Layers
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=5
        )

        self.bn1 = nn.BatchNorm1d(
            num_features=32
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5
        )

        self.bn2 = nn.BatchNorm1d(
            num_features=64
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Classification Layers
        # 64 channels × 59 samples after Conv/Pool blocks
        self.fc1 = nn.Linear(
            in_features=1216,
            out_features=128
        )

        self.dropout = nn.Dropout(
            p=DROPOUT_RATE
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=NUM_CLASSES
        )

    def forward(self, x):

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Classification Head
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        # Return raw logits
        return x
    
    def extract_features(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        return x

