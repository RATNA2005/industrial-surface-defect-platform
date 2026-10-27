"""
Proposed Advanced Vision Model Architecture Visualization
=========================================================
Generates an architectural diagram comparing the BaselineDefectCNN (28K params)
against the proposed Transfer Learning ResNet-18 model (11.2M params),
highlighting the residual skip connections, layer freezing policy,
and adapted defect classification head.

Output:
  - docs/figures/proposed_resnet_architecture.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def generate_comparison_diagram(out_path="docs/figures/proposed_resnet_architecture.png"):
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Title
    ax.text(8.0, 8.0, "Proposed Advanced Model Architecture: Pretrained ResNet-18 Transfer Learning",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1d3557")
    ax.text(8.0, 7.55, "Comparison with Baseline Defect CNN & Industrial Transfer Learning Adaptation Protocol",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#457b9d")

    # -------------------------------------------------------------
    # Top Branch: BaselineDefectCNN (Lightweight Scratch Model)
    # -------------------------------------------------------------
    ax.text(0.5, 6.7, "BaselineDefectCNN (28,358 Parameters | Trained from Scratch):", fontsize=10.5, fontweight="bold", color="#264653")
    
    baseline_boxes = [
        ("Input Image\n(3, 200, 200)", 0.5, 5.8, 1.8, 0.9, "#457b9d"),
        ("Conv Stage 1\n16 Ch | 3x3\n(16, 100, 100)", 2.8, 5.8, 1.8, 0.9, "#2a9d8f"),
        ("Conv Stage 2\n32 Ch | 3x3\n(32, 50, 50)", 5.1, 5.8, 1.8, 0.9, "#2a9d8f"),
        ("Conv Stage 3\n64 Ch | 3x3\n(64, 25, 25)", 7.4, 5.8, 1.8, 0.9, "#2a9d8f"),
        ("Global Avg Pool\nFlatten\n(64 Features)", 9.7, 5.8, 1.8, 0.9, "#e76f51"),
        ("Dense Head\nLinear(64->64)\nLinear(64->6)", 12.0, 5.8, 1.8, 0.9, "#e63946"),
        ("6 Logits\n(NEU Defect)", 14.3, 5.8, 1.3, 0.9, "#1d3557")
    ]

    for label, x, y, w, h, color in baseline_boxes:
        rect = patches.FancyBboxPatch((x, y - h/2), w, h, boxstyle="round,pad=0.08",
                                      facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y, label, ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    for i in range(len(baseline_boxes) - 1):
        x1 = baseline_boxes[i][1] + baseline_boxes[i][3] + 0.05
        x2 = baseline_boxes[i+1][1] - 0.05
        ax.annotate('', xy=(x2, 5.8), xytext=(x1, 5.8),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#264653"))

    # Divider line
    ax.axhline(y=4.8, color="#ccc", linestyle="--", linewidth=1.2)

    # -------------------------------------------------------------
    # Bottom Branch: Proposed Pretrained ResNet-18 (Transfer Learning)
    # -------------------------------------------------------------
    ax.text(0.5, 4.3, "Proposed Model: ResNet-18 (11,179,590 Parameters | Pretrained on ImageNet-1K):", fontsize=10.5, fontweight="bold", color="#1d3557")

    resnet_boxes = [
        ("Input Image\n(3, 200, 200)", 0.5, 3.2, 1.8, 1.1, "#457b9d"),
        ("Stem: Conv1\n7x7, s=2 + MaxPool\n(64, 50, 50)", 2.6, 3.2, 1.9, 1.1, "#1d3557"),
        ("Stage 1 (ResBlock)\n2x [3x3, 64 Ch]\nIdentity Shortcuts", 4.8, 3.2, 2.0, 1.1, "#2b2d42"),
        ("Stage 2 (ResBlock)\n2x [3x3, 128 Ch]\nDownsample Shortcut", 7.1, 3.2, 2.0, 1.1, "#2b2d42"),
        ("Stage 3 (ResBlock)\n2x [3x3, 256 Ch]\nDownsample Shortcut", 9.4, 3.2, 2.0, 1.1, "#2b2d42"),
        ("Stage 4 (ResBlock)\n2x [3x3, 512 Ch]\nDownsample Shortcut", 11.7, 3.2, 2.0, 1.1, "#2b2d42"),
        ("Adapted Defect Head\nGAP -> Dropout(0.3)\nLinear(512 -> 6)", 14.0, 3.2, 1.8, 1.1, "#e63946")
    ]

    for label, x, y, w, h, color in resnet_boxes:
        rect = patches.FancyBboxPatch((x, y - h/2), w, h, boxstyle="round,pad=0.08",
                                      facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y, label, ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    for i in range(len(resnet_boxes) - 1):
        x1 = resnet_boxes[i][1] + resnet_boxes[i][3] + 0.05
        x2 = resnet_boxes[i+1][1] - 0.05
        ax.annotate('', xy=(x2, 3.2), xytext=(x1, 3.2),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#1d3557"))

    # Highlight residual identity skip connection formula box
    math_box = patches.FancyBboxPatch((0.5, 0.4), 7.0, 1.5, boxstyle="round,pad=0.1",
                                     facecolor="#f1faee", edgecolor="#1d3557", linewidth=1.5)
    ax.add_patch(math_box)
    ax.text(4.0, 1.45, "Core Mathematical Foundation: Residual Skip Connection", fontsize=9.5, fontweight="bold", ha="center", color="#1d3557")
    ax.text(4.0, 0.95, r"$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$  (Identity Mapping)" + "\n" +
            r"$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$  (Prevents Vanishing Gradients)",
            fontsize=9.5, ha="center", color="#1d3557")

    # Fine-Tuning Strategy Box
    ft_box = patches.FancyBboxPatch((8.0, 0.4), 7.5, 1.5, boxstyle="round,pad=0.1",
                                   facecolor="#edf2f4", edgecolor="#e63946", linewidth=1.5)
    ax.add_patch(ft_box)
    ax.text(11.75, 1.45, "Two-Phase Fine-Tuning Protocol", fontsize=9.5, fontweight="bold", ha="center", color="#e63946")
    ax.text(11.75, 0.95, "Phase 1: Freeze Backbone (Stem + Stages 1-3), Train Head (lr=1e-3, 3 Epochs)\n" +
                         "Phase 2: Differential Fine-Tuning: Backbone (lr=1e-4) + Head (lr=1e-3)\n" +
                         "Loss: CrossEntropyLoss | Optimizer: AdamW | Scheduler: CosineAnnealingLR",
            fontsize=8.5, ha="center", color="#2b2d42")

    plt.tight_layout()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved proposed architecture diagram to: {out_file}")

if __name__ == "__main__":
    generate_comparison_diagram()
