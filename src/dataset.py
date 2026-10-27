import csv
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)}

def compute_train_statistics(train_csv_path):
    """
    Computes per-channel mean and standard deviation strictly from the training partition.
    This guarantees zero data leakage from validation or test partitions.
    """
    print(f"Computing channel statistics from training set: {train_csv_path}")
    image_paths = []
    with open(train_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(row["image_path"])

    assert len(image_paths) == 1260, f"Expected 1260 training images, got {len(image_paths)}"

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0

    to_tensor = T.ToTensor()
    for p in image_paths:
        with Image.open(p) as img:
            img = img.convert('RGB')
            t = to_tensor(img) # Shape: (3, H, W), values in [0, 1]
            channel_sum += t.sum(dim=[1, 2])
            channel_sum_sq += (t ** 2).sum(dim=[1, 2])
            total_pixels += t.shape[1] * t.shape[2]

    mean = channel_sum / total_pixels
    std = torch.sqrt((channel_sum_sq / total_pixels) - (mean ** 2))

    mean_list = [round(float(m), 4) for m in mean]
    std_list = [round(float(s), 4) for s in std]
    print(f"Computed Training Mean: {mean_list}")
    print(f"Computed Training Std:  {std_list}")
    return mean_list, std_list

def get_transforms(mean, std, image_size=(200, 200)):
    """
    Returns separate transformation pipelines for training and evaluation.
    - Training pipeline: Includes domain-appropriate augmentations (flips, slight rotations, minor contrast).
    - Evaluation pipeline: Strictly deterministic (resize, tensor conversion, normalization).
    """
    train_transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    eval_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform

class NEUDefectDataset(Dataset):
    """
    PyTorch Dataset for the NEU Surface Defect Database.
    Loads image paths and labels from a split CSV manifest.
    """
    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.samples = []

        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "image_path": row["image_path"],
                    "class_name": row["class_name"],
                    "class_id": int(row["class_id"])
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]

        with Image.open(image_path) as img:
            image = img.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = sample["class_id"]
        return image, label, sample["class_name"], sample["filename"]

def get_dataloaders(
    splits_dir="data/splits",
    batch_size=32,
    num_workers=0,
    seed=42,
    image_size=(200, 200)
):
    """
    Factory function to build reproducible DataLoaders for train, val, and test partitions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits_path = Path(splits_dir)
    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    test_csv = splits_path / "test.csv"

    # Compute or load pre-calculated training set statistics
    config_path = splits_path / "pipeline_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            mean = cfg["train_normalization"]["mean"]
            std = cfg["train_normalization"]["std"]
    else:
        mean, std = compute_train_statistics(train_csv)

    train_transform, eval_transform = get_transforms(mean, std, image_size=image_size)

    train_dataset = NEUDefectDataset(train_csv, transform=train_transform)
    val_dataset = NEUDefectDataset(val_csv, transform=eval_transform)
    test_dataset = NEUDefectDataset(test_csv, transform=eval_transform)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, mean, std
