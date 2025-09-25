import os
import torch

from src.model.fsrcnn import FSRCNN
from src.model.rsrcnn import RSRCNN
from src.realtime import run_realtime_sr


def main() -> None:
    # Hyperparameters
    upscaling_factor = 2
    d = 56
    s = 12
    m = 4
    model_name = "rsrcnn_x2_cssim"

    # Torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model_dir = os.path.join("models")
    model = RSRCNN(upscaling_factor=upscaling_factor,
                   d=d,
                   s=s,
                   m=m).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pt"), weights_only=False)["model_state_dict"])
    model.eval()

    # Real time
    run_realtime_sr(model, device)


if __name__ == "__main__":
    main()
