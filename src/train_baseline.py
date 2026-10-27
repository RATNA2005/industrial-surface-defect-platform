"""
Baseline CNN Training Pipeline
===============================
Reproducible training script for BaselineDefectCNN on the NEU Surface Defect Database.

Records:
  - Loss function: CrossEntropyLoss
  - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
  - Batch size: 32
  - Epochs: 20
  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
  - Early stopping: Monitor validation loss (patience=5)
  - Hardware: CPU
  - Checkpoint: models/baseline_cnn_best.pth
"""

import sys
import time
import json
import random
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import get_dataloaders
from src.baseline_cnn import BaselineDefectCNN

def set_global_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc

def train_baseline_model(
    splits_dir="data/splits",
    models_dir="models",
    log_file="docs/baseline_training_log.json",
    epochs=20,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-4,
    seed=42
):
    print("=" * 70)
    print("  PHASE I: BASELINE DEFECT CNN TRAINING (STEP 17)")
    print("=" * 70)

    set_global_seeds(seed)
    device = torch.device("cpu")
    print(f"Execution Device:      {device}")
    print(f"Reproducibility Seed:  {seed}")
    print(f"Batch Size:            {batch_size}")
    print(f"Max Epochs:            {epochs}")
    print(f"Initial Learning Rate: {lr}")
    print(f"Weight Decay:          {weight_decay}")

    # Build DataLoaders
    train_loader, val_loader, test_loader, mean, std = get_dataloaders(
        splits_dir=splits_dir,
        batch_size=batch_size,
        seed=seed
    )
    print(f"Dataset Partitions:    Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")

    # Instantiate Baseline Model
    model = BaselineDefectCNN(in_channels=3, num_classes=6, dropout_rate=0.25).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters:      {total_params:,}")

    # Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = models_path / "baseline_cnn_best.pth"

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    training_history = {
        "config": {
            "model": "BaselineDefectCNN",
            "parameters": total_params,
            "loss_function": "CrossEntropyLoss",
            "optimizer": "Adam",
            "initial_lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "seed": seed,
            "device": str(device)
        },
        "epochs": []
    }

    start_total_time = time.time()
    print("\nStarting Training Loop:\n")
    print("Epoch | Train Loss | Train Acc |  Val Loss  |  Val Acc  | Current LR | Time")
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_acc, 2),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_acc, 2),
            "lr": current_lr,
            "time_seconds": round(elapsed, 2)
        }
        training_history["epochs"].append(epoch_record)

        # Check for checkpoint saving
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            is_best = True
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": training_history["config"]
            }, best_checkpoint_path)
            best_tag = " <-- BEST"
        else:
            patience_counter += 1
            best_tag = ""

        print(f"{epoch:5d} |   {train_loss:.5f}  |   {train_acc:5.1f}%  |   {val_loss:.5f}  |   {val_acc:5.1f}%  |   {current_lr:.5f}  | {elapsed:4.1f}s{best_tag}")

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING TRIGGERED] Validation loss has not improved for {patience} epochs.")
            break

    total_training_time = time.time() - start_total_time
    training_history["summary"] = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "best_val_accuracy": round(best_val_acc, 2),
        "total_training_time_seconds": round(total_training_time, 2),
        "checkpoint_saved": str(best_checkpoint_path).replace("\\", "/")
    }

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best Epoch:           {best_epoch}")
    print(f"  Best Validation Loss: {best_val_loss:.5f}")
    print(f"  Best Validation Acc:  {best_val_acc:.2f}%")
    print(f"  Total Training Time:  {total_training_time:.1f} seconds ({total_training_time/60:.1f} minutes)")
    print(f"  Model Saved To:       {best_checkpoint_path}")
    print("=" * 70 + "\n")

    # Save log to JSON
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2)
    print(f"Saved training log to: {log_path}")

    return training_history

if __name__ == "__main__":
    train_baseline_model()
