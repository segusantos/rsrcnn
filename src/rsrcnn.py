import torch
from torch import nn


class RSRCNN(nn.Module):
    """
    Residual Super-Resolution Convolutional Network
    """

    def __init__(self,
                 upscaling_factor: int,
                 d: int,  # LR feature dimension
                 s: int,  # Number of shrinking filters
                 m: int,  # Mapping depth
                 ) -> None:
        super().__init__()

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, d, 5, padding="same"),
            nn.BatchNorm2d(d),
            nn.PReLU(d)
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, 1),
            nn.BatchNorm2d(s),
            nn.PReLU(s)
        )

        # Non-linear mapping with residual blocks
        self.non_linear_mapping = nn.ModuleList([
            ResidualBlock(s) for _ in range(m)
        ])

        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, 1),
            nn.BatchNorm2d(d),
            nn.PReLU(d)
        )

        # Deconvolution
        self.deconvolution = nn.ConvTranspose2d(
            d, 1, upscaling_factor * 2,
            stride=upscaling_factor,
            padding=upscaling_factor // 2
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.deconvolution.weight, std=1e-3)
        nn.init.zeros_(self.deconvolution.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        features = self.feature_extraction(input)

        # Shrinking
        shrunk_features = self.shrinking(features)

        # Non-linear mapping
        mapped_features = shrunk_features
        for block in self.non_linear_mapping:
            mapped_features = block(mapped_features)

        mapped_features = mapped_features + shrunk_features

        # Expanding
        expanded_features = self.expanding(mapped_features)
        expanded_features = expanded_features + features

        # Deconvolution
        output = self.deconvolution(expanded_features)
        
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding="same"),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding="same"),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
