import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr


class CSSIMLoss(nn.Module):
    def __init__(self, data_range: float = 255.0) -> None:
        super().__init__()
        self.data_range = data_range

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 1 - ssim(y_pred, y_true, data_range=self.data_range, gaussian_kernel=False)


class NPSNRLoss(nn.Module):
    def __init__(self, data_range: float = 255.0) -> None:
        super().__init__()
        self.data_range = data_range

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return -psnr(y_pred, y_true, data_range=self.data_range)
