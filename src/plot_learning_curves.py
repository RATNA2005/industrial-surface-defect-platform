"""
Baseline Defect CNN Learning Curves Visualization & Convergence Analysis
========================================================================
Parses the training log (docs/baseline_training_log.json) and creates
publication-grade learning curves showing:
  1. Training Loss vs. Validation Loss across 20 epochs
  2. Training Accuracy vs. Validation Accuracy across 20 epochs
  3. Generalization Gap (Val Loss - Train Loss) & LR Decay Dynamics

Saves:
  - docs/figures/learning_curves.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def load_training_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_learning_curves_plot(log_data, out_path="docs/figures/learning_curves.png"):
    epochs_data = log_data["epochs"]
    summary = log_data["summary"]

    epochs = [d["epoch"] for d in epochs_data]
    train_loss = [d["train_loss"] for d in epochs_data]
    val_loss = [d["val_loss"] for d in epochs_data]
    train_acc = [d["train_accuracy"] for d in epochs_data]
    val_acc = [d["val_accuracy"] for d in epochs_data]
    lrs = [d["lr"] for d in epochs_data]
    gen_gap = [v - t for v, t in zip(val_loss, train_loss)]

    best_epoch = summary["best_epoch"]
    best_val_loss = summary["best_val_loss"]
    best_val_acc = summary["best_val_accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # -------------------------------------------------------------
    # Panel 1: Loss Convergence
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'o-', color='#1d3557', linewidth=2.2, label='Train Loss (with Aug & Dropout)')
    ax1.plot(epochs, val_loss, 's--', color='#e63946', linewidth=2.2, label='Validation Loss (Deterministic)')
    
    # Highlight best checkpoint
    ax1.scatter([best_epoch], [best_val_loss], color='#2a9d8f', s=180, zorder=5, marker='*', edgecolors='black', label=f'Best Checkpoint (Ep {best_epoch}: {best_val_loss:.4f})')
    ax1.axvline(x=best_epoch, color='#2a9d8f', linestyle=':', alpha=0.8, linewidth=1.5)

    # Learning rate step marker
    lr_drop_epoch = None
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i-1]:
            lr_drop_epoch = epochs[i]
            break
    if lr_drop_epoch:
        ax1.annotate(
            f'LR Decay (1e-3 -> 5e-4)\nat Epoch {lr_drop_epoch}',
            xy=(lr_drop_epoch, val_loss[lr_drop_epoch - 1]),
            xytext=(lr_drop_epoch - 6, val_loss[lr_drop_epoch - 1] + 0.45),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
            fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f1faee', edgecolor='#1d3557')
        )

    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Training Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_xticks(epochs[::2])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Accuracy Dynamics
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(epochs, train_acc, 'o-', color='#1d3557', linewidth=2.2, label='Train Accuracy')
    ax2.plot(epochs, val_acc, 's--', color='#2a9d8f', linewidth=2.2, label='Validation Accuracy')
    
    # 85% Phase I target line
    ax2.axhline(y=85.0, color='#e76f51', linestyle='--', linewidth=1.5, label='Phase I Target Threshold (85%)')
    
    # Peak Val Accuracy Marker
    peak_val_acc = max(val_acc)
    peak_epoch = epochs[val_acc.index(peak_val_acc)]
    ax2.scatter([peak_epoch], [peak_val_acc], color='#f4a261', s=160, zorder=5, marker='D', edgecolors='black', label=f'Peak Val Acc (Ep {peak_epoch}: {peak_val_acc:.1f}%)')

    ax2.set_title("Classification Accuracy (%) vs. Epochs", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Training Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_xticks(epochs[::2])
    ax2.set_ylim(30, 100)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Generalization Gap & LR Dynamics
    # -------------------------------------------------------------
    ax3 = axes[2]
    colors = ['#2a9d8f' if g <= 0 else '#e63946' for g in gen_gap]
    ax3.bar(epochs, gen_gap, color=colors, alpha=0.7, edgecolor='black', width=0.6, label='Generalization Gap (Val - Train)')
    ax3.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0)

    # Secondary axis for Learning Rate
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, lrs, 'd-', color='#457b9d', linewidth=2.0, alpha=0.85, label='Learning Rate')
    ax3_twin.set_ylabel("Learning Rate", fontsize=10, color='#457b9d')
    ax3_twin.tick_params(axis='y', labelcolor='#457b9d')
    ax3_twin.set_yscale('log')

    ax3.set_title("Generalization Gap & Learning Rate", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Training Epoch", fontsize=10)
    ax3.set_ylabel("Loss Gap (Val Loss - Train Loss)", fontsize=10)
    ax3.set_xticks(epochs[::2])
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Combine legends for twin axis
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8.5)

    plt.suptitle("BaselineDefectCNN: Learning Curves & Generalization Dynamics (NEU-DET)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves plot to: {out_file}")

if __name__ == "__main__":
    log_file = root_dir / "docs" / "baseline_training_log.json"
    if not log_file.exists():
        raise FileNotFoundError(f"Training log not found at: {log_file}")
    data = load_training_log(log_file)
    generate_learning_curves_plot(data)
