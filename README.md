# RSRCNN

<div align="center">

![Comparison Results](figures/comparison_x2_14.pdf)
![Comparison Results](figures/comparison_x2_3.pdf)

*Visual comparison of super-resolution results showing Low Resolution (LR), Bicubic interpolation, FSRCNN, RSRCNN, and Ground Truth (GT) images*

</div>

This repository contains the implementation of the Residual Super-Resolution Convolutional Neural Network (RSRCNN), as well as the baseline Fast Super-Resolution Convolutional Neural Network (FSRCNN) model, introduced in the paper [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367).

RSRCNN builds upon the FSRCNN architecture by replacing the non-linear mapping layer with a residual block and adding skip connections, improving performance in image super-resolution tasks.

The models are implemented in PyTorch and trained on standard datasets such as DIV2K, T91, General100, and BSD100. The repository includes trained weights for both FSRCNN and RSRCNN models at x2 and x4 upscaling factors.

Moreover, this work explores the impact of different loss functions to the default MSE loss on super-resolution performance. In particular, it introduces Negative PSNR and Complementary SSIM (CSSIM) losses, which are designed to better align the training objective with perceptual image quality and yield improved results in terms of PSNR and SSIM metrics.

## Usage

The repository includes scripts for training both the baseline FSRCNN and the RSRCNN models, evaluating their performance, visualizing results and testing them in real-time using a webcam. The project uses `uv` for dependency management.

- Install the required dependencies:

  ```bash
  uv sync --locked
  ```

- Build a super-resolution dataset for a given set of images, upscaling factor and patch size:

  ```bash
  uv run -m scripts.dataset
  ```

- Train a model:

  ```bash
  uv run -m scripts.train
  ```

- Evaluate model performance on test datasets:

  ```bash
  uv run -m scripts.eval
  ```

- Visualize and plot comparison results:

  ```bash
  uv run -m scripts.plot
  ```

- Test the model in real-time using your webcam:

  ```bash
  uv run -m scripts.realtime
  ```
