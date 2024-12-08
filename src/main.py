import os
import cv2
import torch
from torch import nn
from torch import optim

from data import load_dataset, get_patches
from fsrcnn import FSRCNN
from train import train


def main() -> None:
    # Hyperparameters
    upscaling_factor = 2
    d = 56 # LR feature dimension (56 for best performance, 32 for real-time)
    s = 12 # Number of shrinking filters (12 for best performance, 5 for real-time)
    m = 4  # Mapping depth (4 for best performance, 1 for real-time)
    scales = [0.9, 0.8, 0.7, 0.6]
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    seed = 69
    epochs = 1000
    batch_size = 64
    lr = 1e-3
    criterion = nn.MSELoss()
    optimizer = optim.Adam
    eval_every = 25

    # Torch
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load datasets
    data_dir = os.path.join("..", "data")
    
    t91_lr, t91_gt = load_dataset(os.path.join(data_dir, "train", "T91", f"X{upscaling_factor}"), scales, angles)
    general100_lr, general100_gt = load_dataset(os.path.join(data_dir, "train", "General100", f"X{upscaling_factor}"), scales, angles)
    x_train, y_train = get_patches(t91_lr + general100_lr, t91_gt + general100_gt, 64)
    x_train = [torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0) for img in x_train]
    y_train = [torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0) for img in y_train]
                               
    bsd100_lr, bsd100_gt = load_dataset(os.path.join(data_dir, "validation", "BSD100", f"X{upscaling_factor}"))
    x_val = [torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0) for img in bsd100_lr]
    y_val = [torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0) for img in bsd100_gt]

    # Model
    model = FSRCNN(upscaling_factor=upscaling_factor,
                   d=d,
                   s=s,
                   m=m).to(device)
    train(model=model,
          x_train=x_train,
          y_train=y_train,
          x_val=x_val,
          y_val=y_val,
          criterion=criterion,
          optimizer=optimizer,
          lr=lr,
          epochs=epochs,
          batch_size=batch_size,
          eval_every=eval_every)


if __name__ == "__main__":
    main()
