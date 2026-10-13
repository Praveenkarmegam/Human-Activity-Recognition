# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
import torch.nn.functional as F


class HAR_CNN(nn.Module):
    """
    1D Convolutional Neural Network for Human Activity Recognition (HAR)
    using 9-axis inertial sensor signals (accelerometer & gyroscope) with 128 time steps.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Conv1d(
            in_channels=9,
            out_channels=32,
            kernel_size=7,
            padding=3
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Global average pooling across time steps
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head with dropout
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, 9, 128)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)
        x = self.fc(x)

        return x
