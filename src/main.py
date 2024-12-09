import os
import cv2
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data import SRDataset
from fsrcnn import FSRCNN
from rsrcnn import RSRCNN
from train import train


def main() -> None:
    # Hyperparameters
    upscaling_factor = 2
    d = 56
    s = 12
    m = 4
    seed = 69
    epochs = 1000
    batch_size = 128
    lr = 1e-3
    criterion = nn.MSELoss()
    optimizer = optim.Adam
    eval_every = 25
    patch_size = 64
    model_name = "best_model_big_x2"
    # train_datasets_names = ["DIV2K"]
    train_datasets_names = ["T91", "General100"]
    val_datasets_names = ["BSD100"]

    # Torch
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    data_dir = os.path.join("..", "data")
    
    train_datasets_dirs = [os.path.join(data_dir, "train", dataset, f"X{upscaling_factor}_PS{patch_size}") for dataset in train_datasets_names]
    train_dataset = SRDataset(train_datasets_dirs, upscaling_factor, device)

    val_datasets_dirs = [os.path.join(data_dir, "validation", dataset, f"X{upscaling_factor}") for dataset in val_datasets_names]
    val_dataset = SRDataset(val_datasets_dirs, upscaling_factor, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = FSRCNN(upscaling_factor=upscaling_factor,
                   d=d,
                   s=s,
                   m=m).to(device)
    print(model)
    print(f"Device: {device}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load model
    model_dir = os.path.join("..", "models")
    if os.path.exists(os.path.join(model_dir, f"{model_name}.pt")):
        checkpoint = torch.load(os.path.join(model_dir, f"{model_name}.pt"), weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded")

    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          criterion=criterion,
          optimizer=optimizer,
          lr=lr,
          epochs=epochs,
          eval_every=eval_every)


if __name__ == "__main__":
    main()
