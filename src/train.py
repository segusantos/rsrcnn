import os
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


@torch.no_grad()
def estimate_loss(model: nn.Module,
                  data_loader: DataLoader,
                  criterion: nn.Module) -> tuple[float, float, float]:
    """
    Estimate the loss of the model on the given data.
    :param model (nn.Module): The model to evaluate.
    :param data_loader (DataLoader): The data loader.
    :param criterion (nn.Module): The loss function.
    :return (tuple[float, float, float]): The loss, PSNR, and SSIM.
    """
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_samples = 0

    for x_batch, y_batch in data_loader:
        y_pred = model(x_batch)
        total_loss += criterion(y_pred, y_batch).item() * len(x_batch)
        for pred, target in zip(y_pred.clamp(0, 255).round(), y_batch):
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            total_psnr += psnr(pred, target, data_range=255.0).item()
            total_ssim += ssim(pred, target, data_range=255.0, gaussian_kernel=False).item()
        total_samples += len(x_batch)

    model.train()
    return total_loss/total_samples, total_psnr/total_samples, total_ssim/total_samples


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          lr: float,
          epochs: int,
          eval_every: int = 10,
          model_name: str = "best_model") -> None:
    """
    Train the model with the given hyperparameters.
    :param model (nn.Module): The model to train.
    :param train_loader (DataLoader): The training data loader.
    :param val_loader (DataLoader): The validation data loader.
    :param criterion (nn.Module): The loss function.
    :param optimizer (optim.Optimizer): The optimizer.
    :param lr (float): The learning rate.
    :param epochs (int): The number of epochs.
    :param eval_every (int): The number of epochs to evaluate the model.
    :param model_name (str): The name of the model to save.
    """
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    metrics = {"train_loss": [], "val_loss": [], "train_psnr": [],
               "val_psnr": [], "train_ssim": [], "val_ssim": []}

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Training loop
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs-1:
            train_loss, train_psnr, train_ssim = estimate_loss(
                model, train_loader, criterion)
            val_loss, val_psnr, val_ssim = estimate_loss(
                model, val_loader, criterion)

            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["train_psnr"].append(train_psnr)
            metrics["train_ssim"].append(train_ssim)
            metrics["val_psnr"].append(val_psnr)
            metrics["val_ssim"].append(val_ssim)

            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_psnr": f"{train_psnr:.2f}",
                "train_ssim": f"{train_ssim:.2f}",
                "val_loss": f"{val_loss:.4f}",
                "val_psnr": f"{val_psnr:.2f}",
                "val_ssim": f"{val_ssim:.2f}",
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                }, os.path.join("models", f"{model_name}.pt"))

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, os.path.join("models", f"{model_name}.pt"))
    print("Training completed.")
