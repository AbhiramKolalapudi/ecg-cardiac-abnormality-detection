import torch.nn as nn

from src.config.constants import NUM_CLASSES
from src.config.constants import DROPOUT_RATE


class DeeperCNN(nn.Module):

    def __init__(self):
        super().__init__()

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

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5
        )

        self.bn3 = nn.BatchNorm1d(
            num_features=128
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=896,
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

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

