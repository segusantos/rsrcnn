import torch
from torch import nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


@torch.no_grad()
def eval_sr_model(model: nn.Module,
                  x: list[torch.Tensor],
                  y: list[torch.Tensor],
                  criterion: nn.Module) -> tuple[list[np.ndarray], float, float, float]:
    """
    Evaluate the model on the given data.
    :param model (nn.Module): The model to evaluate.
    :param x (list[torch.Tensor]): The input data.
    :param y (list[torch.Tensor]): The target data.
    :param criterion (nn.Module): The loss function.
    :return (tuple[list[np.ndarray], float, float, float]): The predicted images, loss, PSNR, and SSIM.
    """
    model.eval()
    loss = 0
    psnr_sum = 0
    ssim_sum = 0
    y_preds = []
    for i in range(0, len(x)):
        x[i] = x[i].unsqueeze(0)
        y[i] = y[i].unsqueeze(0)
        y_pred = model(x[i])
        loss += criterion(y_pred, y[i]).item()
        y_pred_cpu = y_pred.clamp(0, 255).round().cpu().numpy().astype("uint8")[0, 0, :, :]
        y_cpu = y[i].cpu().numpy().astype("uint8")[0, 0, :, :]
        psnr_sum += psnr(y_cpu, y_pred_cpu, data_range=255.0)
        ssim_sum += ssim(y_cpu, y_pred_cpu, data_range=255.0)
        y_preds.append(y_pred_cpu)
    model.train()
    return y_preds, loss/len(x), psnr_sum/len(x), ssim_sum/len(x)
