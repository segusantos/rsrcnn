import os
import cv2
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


def main() -> None:
    plt.style.use(os.path.join("figures", "style.mplstyle"))
    eval_dir = os.path.join("data", "eval")
    upscaling_factor = 2
    image_number = 3 # 14

    lr = cv2.imread(os.path.join(eval_dir, f"lr_x{upscaling_factor}_{image_number}.png"))
    bicubic = cv2.imread(os.path.join(eval_dir, f"bicubic_x{upscaling_factor}_{image_number}.png"))
    fsrcnn = cv2.imread(os.path.join(eval_dir, f"fsrcnn_x{upscaling_factor}_cssim_{image_number}.png"))
    rsrcnn = cv2.imread(os.path.join(eval_dir, f"rsrcnn_x{upscaling_factor}_cssim_{image_number}.png"))
    gt = cv2.imread(os.path.join(eval_dir, f"gt_{image_number}.png"))

    gt_y = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    bicubic_y = cv2.cvtColor(bicubic, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    fsrcnn_y = cv2.cvtColor(fsrcnn, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    rsrcnn_y = cv2.cvtColor(rsrcnn, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    bicubic_psnr = psnr(gt_y, bicubic_y, data_range=255.0)
    bicubic_ssim = ssim(gt_y, bicubic_y, data_range=255.0)
    fsrcnn_psnr = psnr(gt_y, fsrcnn_y, data_range=255.0)
    fsrcnn_ssim = ssim(gt_y, fsrcnn_y, data_range=255.0)
    rsrcnn_psnr = psnr(gt_y, rsrcnn_y, data_range=255.0)
    rsrcnn_ssim = ssim(gt_y, rsrcnn_y, data_range=255.0)

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

    imgs = [gt, bicubic, fsrcnn, rsrcnn]
    titles = [
        "Ground Truth | PSNR | SSIM",
        f"Bicubic | {bicubic_psnr:.2f} dB | {bicubic_ssim:.4f}",
        f"FSRCNN | {fsrcnn_psnr:.2f} dB | {fsrcnn_ssim:.4f}",
        f"RSRCNN | {rsrcnn_psnr:.2f} dB | {rsrcnn_ssim:.4f}",
    ]

    for ax, img, title in zip(axs, imgs, titles):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(title, pad=8)

        # Define crop size (25% of the smaller image dimension, with a sensible minimum)
        h, w = rgb.shape[:2]
        crop_size = int(max(16, min(h, w) // 8))

        # Choose a different part of the image for the zoom (center here)
        x0 = int(100 + w // 2 - crop_size // 2)  # -40
        y0 = int(-80 + h // 2 - crop_size // 2) # -10

        # Draw rectangle on the main image showing where the zoom comes from
        rect = Rectangle((x0, y0), crop_size, crop_size, edgecolor="red", facecolor="none", linewidth=1)
        ax.add_patch(rect)

        # Add an inset showing the zoomed region (kept in the lower-right of the subplot)
        iax = inset_axes(ax, width="30%", height="30%", loc="lower right",
                 bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0.0)
        iax.imshow(rgb[y0 : y0 + crop_size, x0 : x0 + crop_size])

        # hide ticks but keep the axis frame so we can show a border
        iax.set_xticks([])
        iax.set_yticks([])

        # style the inset border (match red rectangle on main image)
        for spine in iax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.0)

        # ensure inset background is transparent
        iax.patch.set_alpha(0.0)

    plt.savefig(os.path.join("figures", f"comparison_x{upscaling_factor}_{image_number}.png"), bbox_inches="tight")
    plt.savefig(os.path.join("figures", f"comparison_x{upscaling_factor}_{image_number}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
