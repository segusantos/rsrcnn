import os
import cv2
import numpy as np
import torch
from torch import nn

from fsrcnn import FSRCNN


def load_dataset(dataset_dir: str) -> list[np.ndarray]:
    """
    Load the dataset from the given directory.
    :param dataset_dir (str): The path to the dataset directory.
    """
    images = []
    for file in os.listdir(dataset_dir):
        image = cv2.imread(os.path.join(dataset_dir, file))
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        images.append(image_ycrcb)
    return images


@torch.no_grad()
def evaluate_model(model: nn.Module,
                  x: list[torch.Tensor]) -> list[np.ndarray]:
    """
    Evaluate the model on the given data.
    :param model (nn.Module): The model to evaluate.
    :param x (list[torch.Tensor]): The input data.
    :return (list[np.ndarray]): The predicted images.
    """
    model.eval()
    y_preds = []
    for i in range(0, len(x)):
        x[i] = x[i].unsqueeze(0)
        y_pred = model(x[i])
        y_pred = y_pred.cpu().numpy().clip(0, 255).astype("uint8")[0, 0, :, :]
        y_preds.append(y_pred)
    model.train()
    return y_preds


def main() -> None:
    # Hyperparameters
    upscaling_factor = 2
    d = 56 # LR feature dimension (56 for best performance, 32 for real-time)
    s = 12 # Number of shrinking filters (12 for best performance, 5 for real-time)
    m = 4  # Mapping depth (4 for best performance, 1 for real-time)
    model_name = "best_model_big_x2"

    # Torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    data_dir = os.path.join("..", "data")
    
    dataset = load_dataset(os.path.join(data_dir, "samples"))
    x = [torch.tensor(img[:, :, 0], dtype=torch.float32, device=device).unsqueeze(0) for img in dataset]

    # Load model
    model_dir = os.path.join("..", "models")

    model = FSRCNN(upscaling_factor=upscaling_factor,
                   d=d,
                   s=s,
                   m=m).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pt"), weights_only=False)["model_state_dict"])

    # Evaluate model
    y_preds = evaluate_model(model, x)

    # Save images
    output_path = os.path.join("..", "data", "predict")
    for i, (lr_image_ycrcb, hr_image_y) in enumerate(zip(dataset, y_preds)):
        bicubic_ycrcb = cv2.resize(lr_image_ycrcb, (lr_image_ycrcb.shape[1]*upscaling_factor, lr_image_ycrcb.shape[0]*upscaling_factor), interpolation=cv2.INTER_CUBIC)
        bicubic_bgr = cv2.cvtColor(bicubic_ycrcb, cv2.COLOR_YCrCb2BGR)
        hr_image_ycrcb = cv2.merge([hr_image_y, bicubic_ycrcb[:, :, 1:]])
        hr_image_bgr = cv2.cvtColor(hr_image_ycrcb, cv2.COLOR_YCrCb2BGR)
        lr_image_bgr = cv2.cvtColor(lr_image_ycrcb, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(output_path, f"bicubic_{i + 1}.png"), bicubic_bgr)
        cv2.imwrite(os.path.join(output_path, f"hr_{i + 1}.png"), hr_image_bgr)
        cv2.imwrite(os.path.join(output_path, f"lr_{i + 1}.png"), lr_image_bgr)


if __name__ == "__main__":
    main()
