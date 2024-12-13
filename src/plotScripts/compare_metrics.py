import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

plt.style.use('../figures.mplstyle')

# Load the uploaded image in grayscale
original_image = cv2.imread('../data/test/Set5/X2/GT/img_001_SRF_2.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded properly
if original_image is None:
    raise ValueError("Image could not be loaded.")

# Generate the noisy image
noise = np.random.normal(0, 34, original_image.shape)  # Gaussian noise with mean=0, std=25
noisy_image = np.clip(original_image + noise, 0, 255).astype(np.uint8)

# Generate the constant-added image
scaled_image = np.clip(original_image * 0.802, 0, 255).astype(np.uint8)

# Plot the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title("\nImagen original\n ", fontsize=30)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title(f"Imagen con ruido gaussiano:\nSSIM={ssim(original_image, noisy_image):.2f},\nMSE={mse(original_image, noisy_image):.2f}", fontsize=25)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(scaled_image, cmap='gray')
plt.title(f"Imagen + Constante:\nSSIM={ssim(original_image, scaled_image):.2f},\nMSE={mse(original_image, scaled_image):.2f}", fontsize=25)
plt.axis('off')

plt.tight_layout()
plt.savefig('./metrics_images.png')
plt.show()
