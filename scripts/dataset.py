import os
import cv2

from src.dataset import build_sr_dataset


def main() -> None:
    dataset_type = "train"
    dataset_name = "General100"
    upscaling_factor = 2
    scales = [0.9, 0.8, 0.7, 0.6]
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    patch_size = 64
    stride = 64

    dataset_dir = os.path.join("data", dataset_type, dataset_name, "original")
    output_dir = os.path.join("data", dataset_type, dataset_name, f"X{upscaling_factor}_P{patch_size}_S{stride}")

    build_sr_dataset(dataset_dir,
                     output_dir,
                     upscaling_factor,
                     scales,
                     angles,
                     patch_size,
                     stride)


if __name__ == "__main__":
    main()
