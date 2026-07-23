import torch
import torch.nn as nn

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x)
        y = y.squeeze(-1)
        y = self.fc(y)
        y = y.unsqueeze(-1)
        return x * y