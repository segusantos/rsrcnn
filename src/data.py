import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def scale_image(image: np.ndarray,
                shapes: list[tuple[int, int]]) -> list[np.ndarray]:
    scaled_images = [image]
    for shape in shapes:
        scaled_image = cv2.resize(image, shape)
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
                  shapes: list[tuple[int, int]],
                  angles: list[int]) -> list[np.ndarray]:
    augmented_images = []
    scaled_images = scale_image(image, shapes)
    for scaled_image in scaled_images:
        rotated_images = rotate_image(scaled_image, angles)
        augmented_images.extend(rotated_images)
    return augmented_images


def get_shapes(lr_shape: np.ndarray,
               gt_shape: np.ndarray,
               scales: list[float]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    find_multiple = lambda x, upscaling_factor: round(x / upscaling_factor) * upscaling_factor
    upscaling_factor = gt_shape[0] // lr_shape[0]
    lr_shapes = []
    gt_shapes = []
    for scale in scales:
        gt_shape = find_multiple(gt_shape[0] * scale, upscaling_factor), find_multiple(gt_shape[1] * scale, upscaling_factor)
        lr_shape = gt_shape[0] // upscaling_factor, gt_shape[1] // upscaling_factor
        gt_shapes.append(gt_shape)
        lr_shapes.append(lr_shape)
    return lr_shapes, gt_shapes


def load_dataset(dataset_dir: str,
                 scales: list[float] = [],
                 angles: list[int] = []) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load the dataset from the given directory
        and apply the augmentation to them.
    :param dataset_dir (str): The path to the dataset directory.
    :param scales (list[float]): The scales to apply to the images.
    :param angles (list[int]): The angles to rotate the images.
    """
    lr_images = []
    gt_images = []
    for file in os.listdir(os.path.join(dataset_dir, "LR")):
        lr_image = cv2.imread(os.path.join(dataset_dir, "LR", file))
        gt_image = cv2.imread(os.path.join(dataset_dir, "GT", file))
        lr_y = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        gt_y = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        lr_shapes, gt_shapes = get_shapes(lr_y.shape, gt_y.shape, scales)
        lr_images.extend(augment_image(lr_y, lr_shapes, angles))
        gt_images.extend(augment_image(gt_y, gt_shapes, angles))
    return lr_images, gt_images


def get_patches(lr_images: list[np.ndarray], gt_images: list[np.ndarray], patch_size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    upscsaling_factor = gt_images[0].shape[0] // lr_images[0].shape[0]
    gt_patches = []
    lr_patches = []
    for lr_image, gt_image in zip(lr_images, gt_images):
        for i in range(0, lr_image.shape[0] - patch_size + 1, patch_size):
            for j in range(0, lr_image.shape[1] - patch_size + 1, patch_size):
                lr_patch = lr_image[i:i+patch_size, j:j+patch_size]
                gt_patch = gt_image[i*upscsaling_factor:i*upscsaling_factor+patch_size*upscsaling_factor, j*upscsaling_factor:j*upscsaling_factor+patch_size*upscsaling_factor]
                lr_patches.append(lr_patch)
                gt_patches.append(gt_patch)
    return lr_patches, gt_patches


def load_eval_dataset(dataset_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load the dataset from the given directory.
    :param dataset_dir (str): The path to the dataset directory.
    """
    lr_images = []
    gt_images = []
    for file in os.listdir(os.path.join(dataset_dir, "LR")):
        lr_image = cv2.imread(os.path.join(dataset_dir, "LR", file))
        gt_image = cv2.imread(os.path.join(dataset_dir, "GT", file))
        lr_ycrcb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)
        gt_ycrcb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)
        lr_images.append(lr_ycrcb)
        gt_images.append(gt_ycrcb)
    return lr_images, gt_images


def load_image(dataset_dir: str,
               image_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the image from the given path
        and apply the augmentation to it.
    :param dataset_dir (str): The path to the dataset directory.
    :param image_path (str): The path to the image.
    :return (np.ndarray, np.ndarray): The low-resolution and ground-truth images.
    """
    lr_image = cv2.imread(os.path.join(dataset_dir, "LR", image_name))
    gt_image = cv2.imread(os.path.join(dataset_dir, "GT", image_name))
    lr_y = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    gt_y = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return lr_y, gt_y


class SRDataset(Dataset):
    def __init__(self,
                 dataset_dirs: list[str],
                 upscaling_factor: int,
                 device: str = "cuda") -> None:
        self.dataset_dirs = dataset_dirs
        self.upscaling_factor = upscaling_factor
        self.device = device
        self.file_list = [(dataset_dir, file_name)  for dataset_dir in dataset_dirs
                                                    for file_name in os.listdir(os.path.join(dataset_dir, "LR"))]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        lr_image, gt_image = load_image(self.file_list[idx][0], self.file_list[idx][1])       
        lr_tensor = torch.from_numpy(lr_image).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_image).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        return lr_tensor, gt_tensor
            

def main() -> None:
    # Hyperparameters
    dataset_type = "train"
    dataset_name = "DIV2K"
    upscaling_factor = 4
    scales = [0.9, 0.8, 0.7, 0.6]
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    patch_size = 240

    # Build and save dataset
    dataset_dir = os.path.join("..", "data", dataset_type, dataset_name, f"X{upscaling_factor}")
    output_dir = os.path.join("..", "data", dataset_type, dataset_name, f"X{upscaling_factor}_PS{patch_size}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "LR"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "GT"), exist_ok=True)

    patch_counter = 0
    for file in tqdm(os.listdir(os.path.join(dataset_dir, "LR"))):
        # Load single image pair
        lr_image = cv2.imread(os.path.join(dataset_dir, "LR", file))
        gt_image = cv2.imread(os.path.join(dataset_dir, "GT", file))
        lr_y = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        gt_y = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        # Augment single image
        lr_shapes, gt_shapes = get_shapes(lr_y.shape, gt_y.shape, scales)
        lr_augmented = augment_image(lr_y, lr_shapes, angles)
        gt_augmented = augment_image(gt_y, gt_shapes, angles)

        # Get and save patches for this image
        lr_patches, gt_patches = get_patches(lr_augmented, gt_augmented, patch_size)
        for lr_patch, gt_patch in zip(lr_patches, gt_patches):
            cv2.imwrite(os.path.join(output_dir, "LR", f"patch_{patch_counter}.png"), lr_patch)
            cv2.imwrite(os.path.join(output_dir, "GT", f"patch_{patch_counter}.png"), gt_patch)
            patch_counter += 1


if __name__ == "__main__":
    main()
