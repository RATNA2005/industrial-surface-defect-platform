"""
Industrial Surface Defect Inspection Platform - Phase I
Step 23: Innovation Architecture Visualization
Generates an architectural diagram of the proposed SE-BaselineDefectCNN with Squeeze-and-Excitation Channel Attention.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_innovation_diagram():
    fig = plt.figure(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Main Title
    plt.suptitle("Innovation Architecture: SE-BaselineDefectCNN\nSqueeze-and-Excitation Channel Attention for Steel Surface Defect Discrimination",
                 fontsize=15, fontweight='bold', color='#111827', y=0.97)

    # ----------------------------------------------------
    # Subplot 1: Squeeze-and-Excitation (SE) Block Mechanism
    # ----------------------------------------------------
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 50)
    ax1.axis('off')
    ax1.set_title("(A) Squeeze-and-Excitation (SE) Recalibration Unit (Hu et al., CVPR 2018)", 
                  fontsize=12, fontweight='bold', loc='left', pad=10)

    # Box styling helper
    def draw_box(ax, x, y, w, h, text, subtext, facecolor, edgecolor='#374151', textcolor='#111827'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                      linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.6, text, ha='center', va='center', fontsize=9.5, fontweight='bold', color=textcolor)
        if subtext:
            ax.text(x + w/2, y + h*0.25, subtext, ha='center', va='center', fontsize=7.5, color='#4b5563', style='italic')

    # Arrow helper
    def draw_arrow(ax, x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563eb"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2, label, ha='center', va='bottom', fontsize=8, color="#1d4ed8", fontweight='semibold')

    # SE Flow nodes
    draw_box(ax1, 2, 18, 14, 14, "Input Feat Map U", "[H x W x C]", "#e0f2fe", "#0284c7")
    draw_arrow(ax1, 17, 25, 23, 25, "F_sq (GAP)")

    draw_box(ax1, 23, 18, 14, 14, "Squeeze Vector z", "[1 x 1 x C]", "#fef3c7", "#d97706")
    draw_arrow(ax1, 38, 25, 44, 25, "W1 [C x C/r]")

    draw_box(ax1, 44, 18, 14, 14, "Bottleneck + ReLU", "[1 x 1 x C/r]", "#fce7f3", "#db2777")
    draw_arrow(ax1, 59, 25, 65, 25, "W2 [C/r x C]")

    draw_box(ax1, 65, 18, 14, 14, "Excitation s", "[1 x 1 x C] (Sigmoid)", "#dcfce7", "#16a34a")
    draw_arrow(ax1, 80, 25, 86, 25, "F_scale (x)")

    draw_box(ax1, 86, 18, 13, 14, "Recalibrated U~", "[H x W x C]", "#ede9fe", "#7c3aed")

    # Residual skip connection for scale
    ax1.plot([9, 9, 92.5, 92.5], [17, 8, 8, 17], color='#9333ea', lw=1.5, linestyle='--')
    ax1.text(50, 5.5, "Channel-wise Recalibration: U~_c = s_c * U_c (Dynamic Feature Scaling)", 
             ha='center', va='center', fontsize=9, color='#6b21a8', fontweight='bold',
             bbox=dict(boxstyle="square,pad=0.3", fc="#f3e8ff", ec="#c084fc", lw=1))

    # Mathematical Annotation
    ax1.text(50, 42, r"$\mathbf{z} = \frac{1}{H \times W}\sum_{i=1}^H \sum_{j=1}^W \mathbf{u}_c(i,j)$  $\longrightarrow$  $\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{z})\right)$  $\longrightarrow$  $\tilde{\mathbf{x}}_c = s_c \cdot \mathbf{u}_c$",
             ha='center', va='center', fontsize=11, color='#1e293b',
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#cbd5e1", lw=1))

    # ----------------------------------------------------
    # Subplot 2: Complete SE-BaselineDefectCNN Pipeline
    # ----------------------------------------------------
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 50)
    ax2.axis('off')
    ax2.set_title("(B) Integration of SE Blocks into BaselineDefectCNN (Parameter Budget: 31,046 [+9.5%])", 
                  fontsize=12, fontweight='bold', loc='left', pad=10)

    stages = [
        ("Input Strip", "[3, 200, 200]", "#f1f5f9", "#64748b", 2, 22, 10, 15),
        ("Stage 1 Conv", "Conv+BN+ReLU\n16 ch [100x100]", "#e0f2fe", "#0284c7", 15, 22, 12, 15),
        ("SE Block 1", "r=4 (16->4->16)\n+136 params", "#dcfce7", "#16a34a", 29, 22, 11, 15),
        ("Stage 2 Conv", "Conv+BN+ReLU\n32 ch [50x50]", "#e0f2fe", "#0284c7", 42, 22, 12, 15),
        ("SE Block 2", "r=4 (32->8->32)\n+528 params", "#dcfce7", "#16a34a", 56, 22, 11, 15),
        ("Stage 3 Conv", "Conv+BN+ReLU\n64 ch [25x25]", "#e0f2fe", "#0284c7", 69, 22, 12, 15),
        ("SE Block 3", "r=4 (64->16->64)\n+2,080 params", "#dcfce7", "#16a34a", 83, 22, 11, 15),
    ]

    for name, sub, fc, ec, x, y, w, h in stages:
        draw_box(ax2, x, y, w, h, name, sub, fc, ec)

    # Connections between stages
    coords = [12, 27, 40, 54, 67, 81]
    targets = [15, 29, 42, 56, 69, 83]
    for c, t in zip(coords, targets):
        draw_arrow(ax2, c, 29.5, t, 29.5)

    # Output Dense Head
    draw_arrow(ax2, 94, 29.5, 96, 29.5)
    ax2.plot([96, 96, 50, 50], [29.5, 14, 14, 10], color='#475569', lw=1.5, linestyle=':')
    
    draw_box(ax2, 35, 1, 30, 9, "GAP + Classifier Head", "Linear(64->64) + Dropout(0.2) + Linear(64->6)\n4,614 parameters", "#fef3c7", "#d97706")

    # Parameter comparison summary card
    summary_text = (
        "Parameter Comparison:\n"
        " - BaselineDefectCNN: 28,358 params (355.7 KB)\n"
        " - SE Attention Overhead: +2,744 params (+9.68%)\n"
        " - SE-BaselineDefectCNN Total: 31,102 params (390 KB)\n"
        " - ResNet-18 (Pretrained): 11,179,590 params (360x larger)\n"
        "Expected Latency: ~7.1 ms (140 FPS on CPU vs 50 ms SLA)"
    )
    ax2.text(82, 6, summary_text, fontsize=8, color='#0f172a',
             bbox=dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#bfdbfe", lw=1))

    plt.tight_layout()
    output_dir = PROJECT_ROOT / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "innovation_se_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved innovation architecture diagram to {output_path}")

if __name__ == "__main__":
    generate_innovation_diagram()
