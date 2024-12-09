import os
import cv2
import numpy as np
import torch
from torch import nn

from data import load_eval_dataset
from fsrcnn import FSRCNN
from rsrcnn import RSRCNN

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


@torch.no_grad()
def evaluate_model(model: nn.Module,
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
        y_pred_ = y_pred.cpu().numpy().clip(0, 255).astype("uint8")[0, 0, :, :]
        y_ = y[i].cpu().numpy().clip(0, 255).astype("uint8")[0, 0, :, :]
        psnr_sum += psnr(y_pred_, y_)
        ssim_sum += ssim(y_pred_, y_)
        y_preds.append(y_pred_)
    model.train()
    return y_preds, loss/len(x), psnr_sum/len(x), ssim_sum/len(x)


def main() -> None:
    # Hyperparameters
    upscaling_factor = 2
    d = 56 # LR feature dimension (56 for best performance, 32 for real-time)
    s = 12 # Number of shrinking filters (12 for best performance, 5 for real-time)
    m = 4  # Mapping depth (4 for best performance, 1 for real-time)
    criterion = nn.MSELoss()
    model_name = "best_model"

    # Torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    data_dir = os.path.join("..", "data")
    
    set5_lr, set5_gt = load_eval_dataset(os.path.join(data_dir, "test", "Set5", f"X{upscaling_factor}"))
    set14_lr, set14_gt = load_eval_dataset(os.path.join(data_dir, "test", "Set14", f"X{upscaling_factor}"))
    eval_lr = set5_lr # + set14_lr
    eval_gt = set5_gt # + set14_gt
    x_test = [torch.tensor(img[:, :, 0], dtype=torch.float32, device=device).unsqueeze(0) for img in eval_lr]
    y_test = [torch.tensor(img[:, :, 0], dtype=torch.float32, device=device).unsqueeze(0) for img in eval_gt]

    # Load model
    model_dir = os.path.join("..", "models")

    model = FSRCNN(upscaling_factor=upscaling_factor,
                   d=d,
                   s=s,
                   m=m).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pt"), weights_only=False)["model_state_dict"])

    # Evaluate model
    criterion = nn.MSELoss()
    y_preds, test_loss, test_psnr, test_ssim = evaluate_model(model, x_test, y_test, criterion)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test PSNR: {test_psnr:.2f}")
    print(f"Test SSIM: {test_ssim:.4f}")

    # Save images
    output_path = os.path.join("..", "data", "eval")
    for i, (lr, gt, pred_y) in enumerate(zip(eval_lr, eval_gt, y_preds)):
        bicubic_ycrcb = cv2.resize(lr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        bicubic_bgr = cv2.cvtColor(bicubic_ycrcb, cv2.COLOR_YCrCb2BGR)
        pred_ycrcb = cv2.merge([pred_y, bicubic_ycrcb[:, :, 1:]])
        pred_bgr = cv2.cvtColor(pred_ycrcb, cv2.COLOR_YCrCb2BGR)
        lr_bgr = cv2.cvtColor(lr, cv2.COLOR_YCrCb2BGR)
        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(output_path, f"bicubic_{i + 1}.png"), bicubic_bgr)
        cv2.imwrite(os.path.join(output_path, f"pred_{i + 1}.png"), pred_bgr)
        cv2.imwrite(os.path.join(output_path, f"lr_{i + 1}.png"), lr_bgr)
        cv2.imwrite(os.path.join(output_path, f"gt_{i + 1}.png"), gt_bgr)


if __name__ == "__main__":
    main()
