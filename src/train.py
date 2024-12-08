import os
from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from skimage.metrics import peak_signal_noise_ratio as psnr
                  

@torch.no_grad()
def estimate_loss(model: nn.Module,
                  x: list[torch.Tensor],
                  y: list[torch.Tensor],
                  criterion: nn.Module,
                  batch_size: int) -> float:  
    """
    Estimate the loss of the model on the given data.
    :param model (nn.Module): The model to evaluate.
    :param x (list[torch.Tensor]): The input data.
    :param y (list[torch.Tensor]): The target data.
    :param criterion (nn.Module): The loss function.
    :param batch_size (int): The batch size.
    :return (float): The estimated loss.
    """
    model.eval()
    loss = 0
    psnr_sum = 0
    indices = torch.randperm(len(x))
    x = [x[i] for i in indices]
    y = [y[i] for i in indices]
    for i in range(0, len(x), batch_size):
        x_batch = torch.stack(x[i:i+batch_size])
        y_batch = torch.stack(y[i:i+batch_size])
        y_pred = model(x_batch)
        loss += criterion(y_pred, y_batch).item()
        psnr_sum += sum(psnr(y_pred[j][0].cpu().numpy().clip(0, 255).astype("uint8"), y_batch[j][0].cpu().numpy().clip(0, 255).astype("uint8")) for j in range(len(y_pred)))
    model.train()
    return loss/len(x), psnr_sum/len(x)


def train(model: nn.Module,
          x_train: list[torch.Tensor],
          y_train: list[torch.Tensor],
          x_val: list[torch.Tensor],
          y_val: list[torch.Tensor],
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          lr: float,
          epochs: int,
          batch_size: int,
          eval_every: int = 10) -> None:
    """
    Train the model with the given hyperparameters.
    :param model (nn.Module): The model to train.
    :param x_train (list[torch.Tensor]): The training input data.
    :param y_train (list[torch.Tensor]): The training target data.
    :param x_val (list[torch.Tensor]): The validation input data.
    :param y_val (list[torch.Tensor]): The validation target data.
    :param criterion (nn.Module): The loss function.
    :param optimizer (optim.Optimizer): The optimizer.
    :param lr (float): The learning rate.
    :param epochs (int): The number of epochs.
    :param batch_size (int): The batch size.
    :param eval_every (int): The number of epochs to evaluate the model.
    """
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    metrics = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}
    
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        indices = torch.randperm(len(x_train))
        x_train = [x_train[i] for i in indices]
        y_train = [y_train[i] for i in indices]
        for i in range(0, len(x_train), batch_size):
            x_batch = torch.stack(x_train[i:i+batch_size])
            y_batch = torch.stack(y_train[i:i+batch_size])            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
        if epoch % eval_every == 0 or epoch == epochs-1:
            train_loss, train_psnr = estimate_loss(model, x_train, y_train, criterion, batch_size)
            val_loss, val_psnr = estimate_loss(model, x_val, y_val, criterion, batch_size)
            
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["train_psnr"].append(train_psnr)
            metrics["val_psnr"].append(val_psnr)
            
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
