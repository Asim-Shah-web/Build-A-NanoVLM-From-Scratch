import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Small CNN image encoder for NanoVLM.

    Input:
        [B, 3, 32, 32]

    Output:
        [B, projection_dim]
    """

    def __init__(
        self,
        projection_dim=64,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Convolutional feature extractor
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv4 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ----------------------------------------------------
        # Projection into shared embedding space
        # ----------------------------------------------------

        self.projection = nn.Linear(
            256,
            projection_dim,
        )

    def forward(self, x):

        # [B, 3, 32, 32]
        x = F.relu(self.conv1(x))

        # [B, 32, 16, 16]
        x = F.relu(self.conv2(x))

        # [B, 64, 8, 8]
        x = F.relu(self.conv3(x))

        # [B, 128, 4, 4]
        x = F.relu(self.conv4(x))

        # [B, 256, 2, 2]
        x = self.global_pool(x)

        # [B, 256, 1, 1]
        x = torch.flatten(x, start_dim=1)

        # [B, 256]
        x = self.projection(x)

        # [B, projection_dim]
        return x