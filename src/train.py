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
                  batch_size: int,
                  eval_batches: int) -> float:  
    """
    Estimate the loss of the model on the given data.
    :param model (nn.Module): The model to evaluate.
    :param x (list[torch.Tensor]): The input data.
    :param y (list[torch.Tensor]): The target data.
    :param criterion (nn.Module): The loss function.
    :param batch_size (int): The batch size.
    :param eval_batches (int): The number of batches to evaluate.
    :return (float): The estimated loss.
    """
    model.eval()
    loss = 0
    psnr_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch, y_batch = crop_batch(x[i:i+batch_size], y[i:i+batch_size])
        y_pred = model(x_batch)
        loss += criterion(y_pred, y_batch).item()
        psnr_cnt += sum(psnr(y_pred[j][0].cpu().numpy().clip(0, 255).astype('uint8'), y_batch[j][0].cpu().numpy().clip(0, 255).astype('uint8')) for j in range(len(y_pred)))
        eval_batches -= 1
        if eval_batches == 0:
            break
    model.train()
    return loss/len(x), psnr_cnt/len(x)
    

def crop_batch(x_batch: list[torch.Tensor], y_batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Crop the batch to the same size.
    :param batch (list[torch.Tensor]): The batch to crop.
    :return (torch.Tensor): The cropped batch.
    """
    upscaling_factor = y_batch[0].shape[-1] // x_batch[0].shape[-1]
    min_height = min(img.shape[-2] for img in y_batch)
    min_width = min(img.shape[-1] for img in y_batch)
    h_indexes = []
    w_indexes = []
    for img in y_batch:
        h, w = img.shape[-2:]
        h_index = torch.randint(0, h-min_height+1, (1,)).item()
        w_index = torch.randint(0, w-min_width+1, (1,)).item()
        h_indexes.append(h_index)
        w_indexes.append(w_index)
    x_batch = torch.stack([img[..., h_index//upscaling_factor:h_index//upscaling_factor+min_height//upscaling_factor,
                               w_index//upscaling_factor:w_index//upscaling_factor+min_width//upscaling_factor] for img, h_index, w_index in zip(x_batch, h_indexes, w_indexes)])
    y_batch = torch.stack([img[..., h_index:h_index+min_height, w_index:w_index+min_width] for img, h_index, w_index in zip(y_batch, h_indexes, w_indexes)])
    return x_batch, y_batch


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
          eval_every: int = 10,
          eval_batches: int = 1) -> None:
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
    :param eval_batches (int): The number of batches to evaluate the model.
    """
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        indices = torch.randperm(len(x_train))
        x_train = [x_train[i] for i in indices]
        y_train = [y_train[i] for i in indices]
        for i in range(0, len(x_train), batch_size):
            x_batch, y_batch = crop_batch(x_train[i:i+batch_size], y_train[i:i+batch_size])
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        if epoch % eval_every == 0 or epoch == epochs-1:
            train_loss, train_psnr = estimate_loss(model, x_train, y_train, criterion, batch_size, eval_batches)
            val_loss, val_psnr = estimate_loss(model, x_val, y_val, criterion, batch_size, eval_batches)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}")
    print("Training completed.")
