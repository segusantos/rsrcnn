import os
import cv2
import torch
from torch import nn

from src.dataloader import load_dataset
from src.model.fsrcnn import FSRCNN
from src.model.rsrcnn import RSRCNN

from src.eval import eval_sr_model


def main() -> None:
    # Hyperparameters
    upscaling_factor = 2
    d = 56  # LR feature dimension (56 for best performance, 32 for real-time)
    s = 12 # Number of shrinking filters (12 for best performance, 5 for real-time)
    m = 4  # Mapping depth (4 for best performance, 1 for real-time)
    criterion = nn.MSELoss()
    model_name = "rsrcnn_x2_cssim"

    # Torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    data_dir = os.path.join("data")

    set5_lr, set5_gt = load_dataset(os.path.join(data_dir, "test", "Set5", f"X{upscaling_factor}"))
    set14_lr, set14_gt = load_dataset(os.path.join(data_dir, "test", "Set14", f"X{upscaling_factor}"))
    eval_lr, eval_gt = set5_lr + set14_lr, set5_gt + set14_gt
    x_test = [torch.tensor(img[:, :, 0], dtype=torch.float32, device=device).unsqueeze(0) / 255.0 for img in eval_lr]
    y_test = [torch.tensor(img[:, :, 0], dtype=torch.float32, device=device).unsqueeze(0) for img in eval_gt]

    # Load model
    model_dir = os.path.join("models")

    model = RSRCNN(upscaling_factor=upscaling_factor,
                   d=d,
                   s=s,
                   m=m).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pt"), weights_only=False)["model_state_dict"])

    # Evaluate model
    criterion = nn.MSELoss()
    y_preds, test_loss, test_psnr, test_ssim = eval_sr_model(model, x_test, y_test, criterion)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test PSNR: {test_psnr:.2f}")
    print(f"Test SSIM: {test_ssim:.4f}")

    # Save images
    output_path = os.path.join("data", "eval")
    for i, (lr, gt, pred_y) in enumerate(zip(eval_lr, eval_gt, y_preds)):
        bicubic_ycrcb = cv2.resize(lr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        bicubic_bgr = cv2.cvtColor(bicubic_ycrcb, cv2.COLOR_YCrCb2BGR)
        pred_ycrcb = cv2.merge([pred_y, bicubic_ycrcb[:, :, 1:]])
        pred_bgr = cv2.cvtColor(pred_ycrcb, cv2.COLOR_YCrCb2BGR)
        lr_bgr = cv2.cvtColor(lr, cv2.COLOR_YCrCb2BGR)
        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(output_path, f"bicubic_x{upscaling_factor}_{i + 1}.png"), bicubic_bgr)
        cv2.imwrite(os.path.join(output_path, f"{model_name}_{i + 1}.png"), pred_bgr)
        cv2.imwrite(os.path.join(output_path, f"lr_x{upscaling_factor}_{i + 1}.png"), lr_bgr)
        cv2.imwrite(os.path.join(output_path, f"gt_{i + 1}.png"), gt_bgr)


if __name__ == "__main__":
    main()
