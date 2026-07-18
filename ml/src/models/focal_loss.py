import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()

        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_weight = (1 - pt) ** self.gamma

        weighted_ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none"
        )

        loss = focal_weight * weighted_ce

        return loss.mean()