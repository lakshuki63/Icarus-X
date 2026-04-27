"""
ICARUS-X — M1 Visionary: AR Feature Head CNN

CNN head that extracts 12-dim AR feature vectors from YOLOv10
detected active region crops. Used when real model is loaded.

Inputs:  Cropped magnetogram patches (64x64)
Outputs: 12-dim AR feature vector per region
"""

import torch
import torch.nn as nn


class ARFeatureHead(nn.Module):
    """CNN feature extractor for active region magnetogram patches."""

    def __init__(self, output_dim: int = 12):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from (B, 1, 64, 64) patches → (B, 12)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.head(x)
