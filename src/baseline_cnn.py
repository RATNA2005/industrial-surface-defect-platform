"""
Baseline Industrial Defect Classifier: Simple CNN Architecture
==============================================================
PyTorch implementation of a lightweight, modular Convolutional Neural Network
for 6-class hot-rolled steel strip surface defect classification.

Architecture Flow:
  Input (3, 200, 200)
    ↓
  Conv Stage 1: Conv2d(3 -> 16, 3x3) + ReLU + MaxPool(2x2) -> (16, 100, 100)
    ↓
  Conv Stage 2: Conv2d(16 -> 32, 3x3) + ReLU + MaxPool(2x2) -> (32, 50, 50)
    ↓
  Conv Stage 3: Conv2d(32 -> 64, 3x3) + ReLU + MaxPool(2x2) -> (64, 25, 25)
    ↓
  Classification Head: AdaptiveAvgPool2d(1x1) + Flatten + Dense(64 -> 64) + Dropout + Dense(64 -> 6)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

class BaselineDefectCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, dropout_rate=0.25):
        super(BaselineDefectCNN, self).__init__()
        
        # -------------------------------------------------------------
        # Stage 1: Low-Level Surface Feature Extraction (Edges, Scratches)
        # Input: (B, 3, 200, 200) -> Output: (B, 16, 100, 100)
        # -------------------------------------------------------------
        self.conv_stage1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # -------------------------------------------------------------
        # Stage 2: Mid-Level Texture & Cavity Detection (Crazing, Pitting)
        # Input: (B, 16, 100, 100) -> Output: (B, 32, 50, 50)
        # -------------------------------------------------------------
        self.conv_stage2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # -------------------------------------------------------------
        # Stage 3: High-Level Defect Morphology (Patches, Rolled-in Scale)
        # Input: (B, 32, 50, 50) -> Output: (B, 64, 25, 25)
        # -------------------------------------------------------------
        self.conv_stage3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # -------------------------------------------------------------
        # Stage 4: Global Aggregation & Decision Classifier Head
        # Input: (B, 64, 25, 25) -> Output: (B, num_classes)
        # -------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # (B, 64, 1, 1)
        self.flatten = nn.Flatten()                     # (B, 64)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=64, out_features=num_classes) # Raw unnormalized logits
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming / He normal initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Standard forward pass returning unnormalized class logits."""
        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

    def inspect_tensor_flow(self, x):
        """
        Diagnostic trace function: prints tensor dimensions after every layer.
        """
        print("\n" + "=" * 70)
        print("  LAYER-BY-LAYER TENSOR FLOW TRACE")
        print("=" * 70)
        print(f"  Input Tensor:                 {list(x.shape)}")
        
        trace = []
        
        # Stage 1
        x1_conv = self.conv_stage1[0](x)
        x1_bn = self.conv_stage1[1](x1_conv)
        x1_relu = self.conv_stage1[2](x1_bn)
        x1_pool = self.conv_stage1[3](x1_relu)
        print(f"  Stage 1 Conv2d (3 -> 16, 3x3): {list(x1_conv.shape)}")
        print(f"  Stage 1 ReLU Activation:      {list(x1_relu.shape)}")
        print(f"  Stage 1 MaxPool2d (2x2):      {list(x1_pool.shape)}")
        trace.append(("Stage 1 MaxPool", list(x1_pool.shape)))

        # Stage 2
        x2_conv = self.conv_stage2[0](x1_pool)
        x2_bn = self.conv_stage2[1](x2_conv)
        x2_relu = self.conv_stage2[2](x2_bn)
        x2_pool = self.conv_stage2[3](x2_relu)
        print(f"  Stage 2 Conv2d (16 -> 32, 3x3):{list(x2_conv.shape)}")
        print(f"  Stage 2 ReLU Activation:      {list(x2_relu.shape)}")
        print(f"  Stage 2 MaxPool2d (2x2):      {list(x2_pool.shape)}")
        trace.append(("Stage 2 MaxPool", list(x2_pool.shape)))

        # Stage 3
        x3_conv = self.conv_stage3[0](x2_pool)
        x3_bn = self.conv_stage3[1](x3_conv)
        x3_relu = self.conv_stage3[2](x3_bn)
        x3_pool = self.conv_stage3[3](x3_relu)
        print(f"  Stage 3 Conv2d (32 -> 64, 3x3):{list(x3_conv.shape)}")
        print(f"  Stage 3 ReLU Activation:      {list(x3_relu.shape)}")
        print(f"  Stage 3 MaxPool2d (2x2):      {list(x3_pool.shape)}")
        trace.append(("Stage 3 MaxPool", list(x3_pool.shape)))

        # Stage 4
        x4_gap = self.global_pool(x3_pool)
        x4_flat = self.flatten(x4_gap)
        logits = self.classifier(x4_flat)
        print(f"  Global Average Pool (1x1):    {list(x4_gap.shape)}")
        print(f"  Flatten Feature Vector:       {list(x4_flat.shape)}")
        print(f"  Dense Classifier Output:      {list(logits.shape)} (6 Defect Logits)")
        print("=" * 70 + "\n")
        return logits

def count_parameters(model):
    """Counts total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def generate_architecture_diagram(figures_dir="docs/figures"):
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    boxes = [
        ("Input Image\n(3, 200, 200)", 0.5, 2.0, 1.8, 1.2, "#457b9d"),
        ("Conv Stage 1\n16 Ch | 3x3\nPool -> (16, 100, 100)", 2.8, 2.0, 2.0, 1.2, "#2a9d8f"),
        ("Conv Stage 2\n32 Ch | 3x3\nPool -> (32, 50, 50)", 5.3, 2.0, 2.0, 1.2, "#2a9d8f"),
        ("Conv Stage 3\n64 Ch | 3x3\nPool -> (64, 25, 25)", 7.8, 2.0, 2.0, 1.2, "#2a9d8f"),
        ("Global Avg Pool\nFlatten\n(64 Features)", 10.3, 2.0, 1.6, 1.2, "#e76f51"),
        ("Dense Head\nLinear(64->64)\nLinear(64->6)", 12.3, 2.0, 1.4, 1.2, "#e63946")
    ]

    for label, x, y, w, h, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y - h/2), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    # Connect with arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + boxes[i][3] + 0.05
        x2 = boxes[i+1][1] - 0.05
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    ax.set_title("Baseline Industrial Defect Inspection CNN Architecture (Phase I)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    out_file = fig_path / "baseline_cnn_architecture.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved baseline CNN architecture visual to: {out_file}")

if __name__ == "__main__":
    model = BaselineDefectCNN(in_channels=3, num_classes=6)
    total_p, trainable_p = count_parameters(model)
    print(f"BaselineDefectCNN instantiated successfully.")
    print(f"Total Parameters:     {total_p:,}")
    print(f"Trainable Parameters: {trainable_p:,}")

    # Run tensor flow trace on a single dummy batch
    dummy_batch = torch.randn(2, 3, 200, 200)
    model.inspect_tensor_flow(dummy_batch)

    # Generate visual diagram
    generate_architecture_diagram()
