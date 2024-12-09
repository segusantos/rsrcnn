import torch
from torch import nn


class FSRCNN(nn.Module):
    """
    Fast Super-Resolution Convolutional Network
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
            nn.PReLU(d)
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, 1),
            nn.PReLU(s)
        )

        # Non-linear mapping
        self.non_linear_mapping = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(s, s, 3, padding="same"),
                nn.PReLU(s)
            ) for _ in range(m)]
        )

        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, 1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.non_linear_mapping(x)
        x = self.expanding(x)
        x = self.deconvolution(x)
        return x
