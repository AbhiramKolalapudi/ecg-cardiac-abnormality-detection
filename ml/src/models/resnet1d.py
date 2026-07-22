import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2
        )

        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2
        )

        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity

        x = self.relu(x)

        return x

class ResNet1D(nn.Module):

    def __init__(self):
        super().__init__()

        # Initial Convolution
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.bn1 = nn.BatchNorm1d(32)

        # Residual Stage 1
        self.layer1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32)
        )

        # Transition to 64 Channels
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )

        self.bn2 = nn.BatchNorm1d(64)

        # Residual Stage 2
        self.layer2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # Common Layers
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=2
        )

        self.global_pool = nn.AdaptiveAvgPool1d(
            output_size=1
        )

        self.flatten = nn.Flatten()

        # Classification Layer
        self.fc = nn.Linear(
            in_features=64,
            out_features=5
        )

    def forward(self, x):

        # Initial Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Residual Stage 1
        x = self.layer1(x)

        # Transition
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
    
        # Residual Stage 2
        x = self.layer2(x)

        # Classification Head
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x 