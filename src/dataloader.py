import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
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
        dataset_dir, file_name = self.file_list[idx]
        lr_image = cv2.imread(os.path.join(dataset_dir, "LR", file_name), cv2.IMREAD_GRAYSCALE)
        gt_image = cv2.imread(os.path.join(dataset_dir, "GT", file_name), cv2.IMREAD_GRAYSCALE)
        lr_tensor = torch.from_numpy(lr_image).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_image).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        return lr_tensor, gt_tensor


class ValDataset(Dataset):
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
        dataset_dir, file_name = self.file_list[idx]
        lr_image = cv2.imread(os.path.join(dataset_dir, "LR", file_name), cv2.IMREAD_COLOR)
        gt_image = cv2.imread(os.path.join(dataset_dir, "GT", file_name), cv2.IMREAD_COLOR)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        lr_image = cv2.resize(gt_image, (gt_image.shape[1] // self.upscaling_factor, gt_image.shape[0] // self.upscaling_factor), cv2.INTER_CUBIC)

        lr_tensor = torch.from_numpy(lr_image).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_image).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        return lr_tensor, gt_tensor


def load_dataset(dataset_dir: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
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
