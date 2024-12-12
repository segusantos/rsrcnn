import os
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
        y_pred_np = y_pred.cpu().numpy()[:, 0].clip(0, 255).astype("uint8")
        y_batch_np = y_batch.cpu().numpy()[:, 0].clip(0, 255).astype("uint8")
        total_psnr += sum(psnr(pred, true) for pred, true in zip(y_pred_np, y_batch_np))
        total_ssim += sum(ssim(pred, true) for pred, true in zip(y_pred_np, y_batch_np))
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
          eval_every: int = 10) -> None:
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
    """
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    metrics = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": [], "train_ssim": [], "val_ssim": []}
    
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
            train_loss, train_psnr, train_ssim = estimate_loss(model, train_loader, criterion)
            val_loss, val_psnr, val_ssim = estimate_loss(model, val_loader, criterion)
            
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["train_psnr"].append(train_psnr)
            metrics["train_ssim"].append(train_ssim)
            metrics["val_psnr"].append(val_psnr)
            metrics["val_ssim"].append(val_ssim)

            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_psnr": f"{train_psnr:.2f}",
                "val_loss": f"{val_loss:.4f}",
                "val_psnr": f"{val_psnr:.2f}"
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                }, os.path.join("..", "models", "best_model.pt"))
                
    torch.save(metrics, "training_metrics.pt")
    print("Training completed.")
