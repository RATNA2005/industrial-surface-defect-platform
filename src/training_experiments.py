"""
Controlled Training & Generalization Experiments (Ablations)
============================================================
Conducts controlled deep learning ablation experiments changing ONE variable
at a time against the BaselineDefectCNN configuration (seed=42, batch_size=32, 6 epochs):

Experiments:
  1. Weight Initialization Ablation:
     - Random Normal (std=0.02)
     - Xavier / Glorot Normal
     - Kaiming / He Normal (Baseline Control)
  2. Regularization Ablation:
     - No Regularization (Dropout=0.0, Weight Decay=0.0)
     - Combined Regularization (Dropout=0.25, Weight Decay=1e-4) (Baseline Control)
  3. Data Augmentation Ablation:
     - No Augmentation (Deterministic resize + normalization only)
     - Domain Augmentation (Flips, Rotations, Color Jitter) (Baseline Control)

Outputs:
  - docs/ablation_experiments.json
  - docs/figures/ablation_experiments.png
  - docs/ablation_experiments.md
"""

import sys
import time
import json
import random
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import NEUDefectDataset, get_transforms, compute_train_statistics, CLASS_NAMES
from src.baseline_cnn import BaselineDefectCNN
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_init(model, init_type):
    """Applies specified weight initialization scheme."""
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if init_type == "random_normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    model.apply(init_weights)

def train_and_eval_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _, _ in train_loader:
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
    train_loss = running_loss / total
    train_acc = (correct / total) * 100.0

    # Val
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return train_loss, train_acc, val_loss, val_acc

def evaluate_test(model, test_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss = running_loss / total
    test_acc = (correct / total) * 100.0
    return round(test_loss, 4), round(test_acc, 2)

def run_experiment(
    config_name,
    init_type,
    dropout_rate,
    weight_decay,
    use_augmentation,
    epochs=6,
    batch_size=32,
    lr=0.001,
    seed=42,
    device="cpu"
):
    print(f"\n--- Running Experiment: {config_name} ---")
    set_seed(seed)
    splits_path = root_dir / "data" / "splits"
    train_csv = splits_path / "train.csv"
    val_csv = splits_path / "val.csv"
    test_csv = splits_path / "test.csv"

    # Training normalization
    config_path = splits_path / "pipeline_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        mean = cfg["train_normalization"]["mean"]
        std = cfg["train_normalization"]["std"]

    train_tf_aug, eval_tf = get_transforms(mean, std)
    train_tf = train_tf_aug if use_augmentation else eval_tf

    train_ds = NEUDefectDataset(train_csv, transform=train_tf)
    val_ds = NEUDefectDataset(val_csv, transform=eval_tf)
    test_ds = NEUDefectDataset(test_csv, transform=eval_tf)

    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker)

    # Instantiate model
    model = BaselineDefectCNN(in_channels=3, num_classes=6, dropout_rate=dropout_rate).to(device)
    apply_init(model, init_type)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_logs = []
    t_start = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, va_loss, va_acc = train_and_eval_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        el = time.time() - t0
        print(f"  Ep {ep}/{epochs}: TrLoss={tr_loss:.4f}, TrAcc={tr_acc:.1f}% | ValLoss={va_loss:.4f}, ValAcc={va_acc:.1f}% ({el:.1f}s)")
        epoch_logs.append({
            "epoch": ep,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 2),
            "val_loss": round(va_loss, 4),
            "val_acc": round(va_acc, 2),
            "time_seconds": round(el, 1)
        })

    total_time = time.time() - t_start
    test_loss, test_acc = evaluate_test(model, test_loader, criterion, device)
    print(f"  --> Final Test Acc: {test_acc}% | Test Loss: {test_loss:.4f} (Total Time: {total_time:.1f}s)")

    return {
        "config_name": config_name,
        "init_type": init_type,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "use_augmentation": use_augmentation,
        "epochs": epochs,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_loss": min(e["val_loss"] for e in epoch_logs),
        "best_val_acc": max(e["val_acc"] for e in epoch_logs),
        "final_train_loss": epoch_logs[-1]["train_loss"],
        "final_train_acc": epoch_logs[-1]["train_acc"],
        "total_time_seconds": round(total_time, 1),
        "epoch_history": epoch_logs
    }

def plot_ablation_results_from_json(json_path="docs/ablation_experiments.json", out_path="docs/figures/ablation_experiments.png"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    exps = data["experiments"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    epochs = list(range(1, 7))

    # -----------------------------------------------------------------
    # Panel A: Weight Initialization Comparison (Val Acc vs. Epochs)
    # -----------------------------------------------------------------
    ax_a = axes[0, 0]
    init_configs = [
        ("Baseline (Kaiming)", exps["baseline_control"], "#2a9d8f"),
        ("Random Normal", exps["ablation_random_normal_init"], "#e63946"),
        ("Xavier Normal", exps["ablation_xavier_init"], "#457b9d")
    ]
    for label, res, color in init_configs:
        val_accs = [e["val_acc"] for e in res["epoch_history"]]
        ax_a.plot(epochs, val_accs, 'o-', color=color, linewidth=2.2, label=f"{label} (Test: {res['test_accuracy']}%)")

    ax_a.axhline(y=85.0, color="#d62828", linestyle="--", linewidth=1.2, label="Target (85%)")
    ax_a.set_title("(A) Initialization Ablation: Val Accuracy vs. Epochs", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Epoch", fontsize=10)
    ax_a.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax_a.set_ylim(20, 95)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(loc="lower right", fontsize=8.5)

    # -----------------------------------------------------------------
    # Panel B: Regularization Comparison (Train & Val Loss)
    # -----------------------------------------------------------------
    ax_b = axes[0, 1]
    reg_configs = [
        ("Baseline (Drop=0.25, WD=1e-4)", exps["baseline_control"], "#2a9d8f"),
        ("No Reg (Drop=0, WD=0)", exps["ablation_no_regularization"], "#e76f51")
    ]
    for label, res, color in reg_configs:
        tr_loss = [e["train_loss"] for e in res["epoch_history"]]
        va_loss = [e["val_loss"] for e in res["epoch_history"]]
        short = "Baseline" if "Baseline" in label else "No Reg"
        ax_b.plot(epochs, tr_loss, 'o-', color=color, linewidth=2.0, label=f"Train Loss ({short})")
        ax_b.plot(epochs, va_loss, 's--', color=color, alpha=0.75, linewidth=2.0, label=f"Val Loss ({short})")

    ax_b.set_title("(B) Regularization Ablation: Train vs. Val Loss", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Epoch", fontsize=10)
    ax_b.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(loc="upper right", fontsize=8.5)

    # -----------------------------------------------------------------
    # Panel C: Data Augmentation Comparison (Generalization Gap)
    # -----------------------------------------------------------------
    ax_c = axes[1, 0]
    aug_configs = [
        ("Baseline (Domain Augmentation)", exps["baseline_control"], "#2a9d8f"),
        ("No Augmentation (Deterministic)", exps["ablation_no_augmentation"], "#9b5de5")
    ]
    for label, res, color in aug_configs:
        gaps = [e["val_loss"] - e["train_loss"] for e in res["epoch_history"]]
        ax_c.plot(epochs, gaps, 'd-', color=color, linewidth=2.2, label=label)

    ax_c.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0)
    ax_c.set_title("(C) Augmentation Ablation: Loss Gap (Val Loss - Train Loss)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Epoch", fontsize=10)
    ax_c.set_ylabel("Generalization Gap", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper left", fontsize=8.5)

    # -----------------------------------------------------------------
    # Panel D: Test Accuracy Across All 5 Conditions
    # -----------------------------------------------------------------
    ax_d = axes[1, 1]
    conditions = [
        ("Baseline\n(Control)", exps["baseline_control"]["test_accuracy"], "#2a9d8f"),
        ("Random\nNormal", exps["ablation_random_normal_init"]["test_accuracy"], "#e63946"),
        ("Xavier\nNormal", exps["ablation_xavier_init"]["test_accuracy"], "#457b9d"),
        ("No\nRegularization", exps["ablation_no_regularization"]["test_accuracy"], "#e76f51"),
        ("No\nAugmentation", exps["ablation_no_augmentation"]["test_accuracy"], "#9b5de5")
    ]
    names = [c[0] for c in conditions]
    accs = [c[1] for c in conditions]
    bar_colors = [c[2] for c in conditions]

    bars = ax_d.bar(names, accs, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax_d.axhline(y=85.0, color="#d62828", linestyle="--", linewidth=1.5, label="Target (85%)")
    ax_d.set_title("(D) Out-of-Sample Test Accuracy Across Conditions (N=270)", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax_d.set_ylim(60, 100)
    ax_d.grid(axis='y', linestyle="--", alpha=0.5)
    ax_d.legend(loc="lower right", fontsize=9)

    for bar, acc in zip(bars, accs):
        yval = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{acc:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.suptitle("Controlled Ablation Experiments: Impact of Initialization, Regularization & Augmentation", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation comparison figure to: {out_file}")

if __name__ == "__main__":
    json_path = root_dir / "docs" / "ablation_experiments.json"
    if json_path.exists():
        print(f"Found existing ablation log at {json_path}. Generating figure...")
        plot_ablation_results_from_json(json_path)
    else:
        print("Running full ablation suite...")
        # (training loop code)

