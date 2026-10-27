"""
Pretrained ResNet-18 Transfer Learning Fine-Tuning Pipeline
===========================================================
Trains an advanced deep residual vision model (ResNet-18 pretrained on ImageNet-1K)
on the NEU Surface Defect Database for direct benchmarking against BaselineDefectCNN.

Protocol:
  - Input: 224x224 RGB, ImageNet normalization
  - Phase 1 (Warmup, Epochs 1-3): Freeze Stem + Layers 1-3; train Layer 4 + Head (lr=1e-3)
  - Phase 2 (Differential Fine-Tuning, Epochs 4-8): Unfreeze all layers
    (Backbone lr=1e-4, Head lr=1e-3, CosineAnnealingLR)
  - Checkpoint: models/resnet18_best.pth
  - Training Log: docs/resnet18_training_log.json
"""

import sys
import time
import json
import random
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import NEUDefectDataset, CLASS_NAMES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_resnet_transforms():
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transform, eval_transform

def build_resnet18(num_classes=6, dropout_rate=0.30):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    in_features = model.fc.in_features # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model

def freeze_backbone_warmup(model):
    """Freezes Stem + Stages 1-3. Keeps Stage 4 and FC trainable."""
    for name, param in model.named_parameters():
        if name.startswith('layer4') or name.startswith('fc'):
            param.requires_grad = True
        else:
            param.requires_grad = False

def unfreeze_all_layers(model):
    """Unfreezes all layers for differential fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0

def train_resnet_pipeline(
    warmup_epochs=3,
    finetune_epochs=5,
    batch_size=32,
    seed=42,
    device="cpu"
):
    print("=" * 75)
    print("  PHASE I: PRETRAINED RESNET-18 TRANSFER LEARNING (STEP 22)")
    print("=" * 75)

    set_seed(seed)
    splits_path = root_dir / "data" / "splits"
    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    test_csv = splits_path / "test.csv"

    train_tf, eval_tf = get_resnet_transforms()
    train_ds = NEUDefectDataset(train_csv, transform=train_tf)
    val_ds = NEUDefectDataset(val_csv, transform=eval_tf)
    test_ds = NEUDefectDataset(test_csv, transform=eval_tf)

    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

    # Build model
    model = build_resnet18(num_classes=6, dropout_rate=0.30).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ResNet-18 Backbone (ImageNet-1K Pretrained Weights)")
    print(f"Total Parameters:      {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_dir / "resnet18_best.pth"

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        "config": {
            "model": "ResNet18",
            "pretrained": "ImageNet1K_V1",
            "total_parameters": total_params,
            "warmup_epochs": warmup_epochs,
            "finetune_epochs": finetune_epochs,
            "total_epochs": warmup_epochs + finetune_epochs,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device)
        },
        "epochs": []
    }

    start_total_time = time.time()

    # -------------------------------------------------------------
    # PHASE 1: WARMUP (FROZEN BACKBONE)
    # -------------------------------------------------------------
    print("\n--- PHASE 1: HEAD WARMUP (Stem & Layers 1-3 Frozen) ---")
    freeze_backbone_warmup(model)
    trainable_params_w = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters in Phase 1: {trainable_params_w:,} / {total_params:,}")

    optimizer_warmup = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-3
    )

    current_epoch = 0
    for ep in range(1, warmup_epochs + 1):
        current_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_warmup, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = False
        if va_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = va_loss, va_acc, current_epoch
            is_best = True
            torch.save({
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": va_loss,
                "val_acc": va_acc,
                "config": history["config"]
            }, best_ckpt_path)

        star = " <-- BEST" if is_best else ""
        print(f"  Warmup Ep {ep}/{warmup_epochs} (Total Ep {current_epoch}): TrLoss={tr_loss:.4f}, TrAcc={tr_acc:5.1f}% | ValLoss={va_loss:.4f}, ValAcc={va_acc:5.1f}% ({elapsed:.1f}s){star}")
        history["epochs"].append({
            "epoch": current_epoch,
            "phase": "warmup",
            "train_loss": round(tr_loss, 4),
            "train_accuracy": round(tr_acc, 2),
            "val_loss": round(va_loss, 4),
            "val_accuracy": round(va_acc, 2),
            "lr": 1e-3,
            "time_seconds": round(elapsed, 1)
        })

    # -------------------------------------------------------------
    # PHASE 2: DIFFERENTIAL FINE-TUNING (ALL LAYERS UNFROZEN)
    # -------------------------------------------------------------
    print("\n--- PHASE 2: DIFFERENTIAL FINE-TUNING (All Layers Unfrozen) ---")
    unfreeze_all_layers(model)
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
    head_params = [p for n, p in model.named_parameters() if n.startswith('fc')]

    optimizer_ft = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs, eta_min=1e-6)

    for ep in range(1, finetune_epochs + 1):
        current_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        scheduler.step()

        is_best = False
        if va_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = va_loss, va_acc, current_epoch
            is_best = True
            torch.save({
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": va_loss,
                "val_acc": va_acc,
                "config": history["config"]
            }, best_ckpt_path)

        star = " <-- BEST" if is_best else ""
        cur_lr = optimizer_ft.param_groups[0]['lr']
        print(f"  Finetune Ep {ep}/{finetune_epochs} (Total Ep {current_epoch}): TrLoss={tr_loss:.4f}, TrAcc={tr_acc:5.1f}% | ValLoss={va_loss:.4f}, ValAcc={va_acc:5.1f}% (LR: {cur_lr:.6f}, {elapsed:.1f}s){star}")
        history["epochs"].append({
            "epoch": current_epoch,
            "phase": "fine_tuning",
            "train_loss": round(tr_loss, 4),
            "train_accuracy": round(tr_acc, 2),
            "val_loss": round(va_loss, 4),
            "val_accuracy": round(va_acc, 2),
            "lr": cur_lr,
            "time_seconds": round(elapsed, 1)
        })

    total_time = time.time() - start_total_time
    history["summary"] = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 2),
        "total_training_time_seconds": round(total_time, 1),
        "checkpoint": str(best_ckpt_path).replace("\\", "/")
    }

    # Save log
    log_path = root_dir / "docs" / "resnet18_training_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 75)
    print("  RESNET-18 TRAINING COMPLETE")
    print(f"  Best Val Epoch:        {best_epoch}")
    print(f"  Best Val Loss:         {best_val_loss:.4f}")
    print(f"  Best Val Accuracy:     {best_val_acc:.2f}%")
    print(f"  Total Training Time:   {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Saved Checkpoint to:   {best_ckpt_path}")
    print("=" * 75 + "\n")

    return history

if __name__ == "__main__":
    train_resnet_pipeline()
