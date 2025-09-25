import os
import numpy as np
import cv2
from tqdm import tqdm


def scale_image(image: np.ndarray,
                scales: list[float]) -> list[np.ndarray]:
    scaled_images = [image]
    for scale in scales:
        scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        scaled_images.append(scaled_image)
    return scaled_images


def rotate_image(image: np.ndarray,
                 angles: list[int]) -> list[np.ndarray]:
    rotated_images = [image]
    for angle in angles:
        rotated_image = cv2.rotate(image, angle)
        rotated_images.append(rotated_image)
    return rotated_images


def augment_image(image: np.ndarray,
                  scales: list[float],
                  angles: list[int]) -> list[np.ndarray]:
    augmented_images = []
    scaled_images = scale_image(image, scales)
    for scaled_image in scaled_images:
        rotated_images = rotate_image(scaled_image, angles)
        augmented_images.extend(rotated_images)
    return augmented_images


def get_patches(images: list[np.ndarray],
                patch_size: int,
                stride: int = -1) -> list[np.ndarray]:
    stride = stride if stride != -1 else patch_size
    patches = []
    for image in images:
        for i in range(0, image.shape[0] - patch_size + 1, stride):
            for j in range(0, image.shape[1] - patch_size + 1, stride):
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
    return patches


def get_lr_patches(patches: list[np.ndarray],
                   upscaling_factor: int) -> list[np.ndarray]:
    lr_patches = []
    for patch in patches:
        assert patch.shape[0] % upscaling_factor == 0 and patch.shape[1] % upscaling_factor == 0
        lr_patch = cv2.resize(patch, (0, 0), fx=1/upscaling_factor, fy=1/upscaling_factor, interpolation=cv2.INTER_AREA)
        lr_patches.append(lr_patch)
    return lr_patches


def build_sr_dataset(dataset_dir: str,
                     output_dir: str,
                     upscaling_factor: int,
                     scales: list[float],
                     angles: list[int],
                     patch_size: int,
                     stride: int) -> None:
    os.makedirs(os.path.join(output_dir, "GT"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "LR"), exist_ok=True)
    patch_counter = 0
    for file in tqdm(os.listdir(dataset_dir)):
        image = cv2.imread(os.path.join(dataset_dir, file), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        augmented_images = augment_image(image, scales, angles)
        patches = get_patches(augmented_images, patch_size, stride)
        lr_patches = get_lr_patches(patches, upscaling_factor)

        for patch, lr_patch in zip(patches, lr_patches):
            cv2.imwrite(os.path.join(output_dir, "GT",f"patch_{patch_counter}.bmp"), patch)
            cv2.imwrite(os.path.join(output_dir, "LR", f"patch_{patch_counter}.bmp"), lr_patch)
            patch_counter += 1
