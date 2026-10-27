import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    compute_train_statistics,
    get_transforms,
    get_dataloaders
)

def test_and_document_pipeline():
    print("Testing Reproducible Data Pipeline...")
    splits_dir = Path("data/splits")
    train_csv = splits_dir / "train.csv"

    # Step 1: Compute training statistics
    mean, std = compute_train_statistics(train_csv)

    # Step 2: Build pipeline configuration record
    config_record = {
        "data_version": "NEU-DET Benchmark v1.0",
        "split_protocol": "Stratified 70/15/15",
        "random_seed": 42,
        "input_dimensions": [3, 200, 200],
        "num_classes": len(CLASS_NAMES),
        "class_mapping": CLASS_TO_IDX,
        "train_normalization": {
            "mean": mean,
            "std": std,
            "note": "Calculated strictly on the 1,260 training images to prevent data leakage"
        },
        "augmentation_policy": {
            "training": [
                "Resize(200, 200)",
                "RandomHorizontalFlip(p=0.5)",
                "RandomVerticalFlip(p=0.5)",
                "RandomRotation(degrees=15)",
                "ColorJitter(brightness=0.1, contrast=0.1)",
                "ToTensor()",
                "Normalize(mean, std)"
            ],
            "validation_and_test": [
                "Resize(200, 200)",
                "ToTensor()",
                "Normalize(mean, std)"
            ],
            "note": "Validation and test sets receive zero random augmentation for deterministic evaluation"
        }
    }

    config_path = splits_dir / "pipeline_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_record, f, indent=2)
    print(f"Saved pipeline configuration to: {config_path}")

    # Step 3: Instantiate DataLoaders and verify batches
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        splits_dir="data/splits",
        batch_size=32,
        num_workers=0,
        seed=42
    )

    print(f"Train batches: {len(train_loader)} (Dataset size: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (Dataset size: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (Dataset size: {len(test_loader.dataset)})")

    # Fetch one batch from each loader to inspect tensors
    for loader_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        images, labels, class_names_batch, filenames = next(iter(loader))
        print(f"[{loader_name} Loader Batch Check]")
        print(f"  Images tensor shape: {images.shape} | dtype: {images.dtype}")
        print(f"  Labels tensor shape: {labels.shape} | dtype: {labels.dtype}")
        print(f"  Min value: {images.min().item():.3f} | Max value: {images.max().item():.3f}")
        assert images.shape == (32, 3, 200, 200), f"Unexpected shape {images.shape}"
        assert labels.shape == (32,), f"Unexpected label shape {labels.shape}"
        assert labels.min().item() >= 0 and labels.max().item() < 6, "Labels out of bounds [0, 5]"

    print("ALL DATALOADER BATCH CHECKS PASSED [OK]")

    # Step 4: Visual Demonstration of Training Augmentations vs Deterministic Validation
    sample_img_path = "data/raw/images/scratches_1.jpg"
    raw_img = Image.open(sample_img_path).convert('RGB')
    train_tf, eval_tf = get_transforms(mean, std, image_size=(200, 200))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    
    # Raw Image
    axes[0].imshow(raw_img)
    axes[0].set_title("Original Raw Image\n(scratches_1.jpg)", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Un-normalize helper for visualization
    mean_arr = np.array(mean)
    std_arr = np.array(std)

    def denormalize(tensor):
        img_np = tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * std_arr) + mean_arr
        return np.clip(img_np, 0.0, 1.0)

    # 3 Augmented Training Variations
    torch.manual_seed(101)
    for i in range(3):
        aug_tensor = train_tf(raw_img)
        axes[i + 1].imshow(denormalize(aug_tensor))
        axes[i + 1].set_title(f"Augmented Train #{i+1}\n(Flip/Rot/Jitter)", fontsize=10, fontweight='bold')
        axes[i + 1].axis('off')

    # 1 Deterministic Validation Processing
    eval_tensor = eval_tf(raw_img)
    axes[4].imshow(denormalize(eval_tensor))
    axes[4].set_title("Deterministic Eval\n(Val/Test Pipeline)", fontsize=10, fontweight='bold')
    axes[4].axis('off')

    plt.suptitle("NEU Surface Defect Pipeline: Augmentation Policy Demonstration", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_fig = Path("docs/figures/pipeline_samples.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved augmentation visualization to: {out_fig}")

    return config_record

if __name__ == "__main__":
    test_and_document_pipeline()
