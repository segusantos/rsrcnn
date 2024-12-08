import os
import numpy as np
import cv2


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
    even = lambda x: round(x / 2) * 2
    upscaling_factor = gt_shape[0] // lr_shape[0]
    lr_shapes = []
    gt_shapes = []
    for scale in scales:
        gt_shape = even(gt_shape[0] * scale), even(gt_shape[1] * scale)
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


if __name__ == "__main__":
    dataset = os.path.join("..", "data", "train", "General100", "X2")
    scales = [0.9, 0.8, 0.7, 0.6]
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE] 
    x_train, y_train = load_dataset(dataset, scales, angles)
    # x_test, y_test = load_dataset(dataset)
    for i, (lr, gt) in enumerate(zip(x_train, y_train)):
        cv2.imwrite(f"../data/aux/lr_{i + 1}.bmp", lr)
        cv2.imwrite(f"../data/aux/gt_{i + 1}.bmp", gt)
