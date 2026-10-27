"""
Industrial Surface Defect Inspection Platform - Phase I
Step 26: Literature & Contribution Network Visualization
Generates a structured visual lineage mapping foundational literature, open-source frameworks, and contribution roles.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_contribution_network():
    fig = plt.figure(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plt.suptitle("Phase I Literature Lineage, Software Stack & Contribution Allocation",
                 fontsize=15, fontweight='bold', color='#111827', y=0.97)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_box(x, y, w, h, title, content, fc, ec, title_color='#111827', title_size=10.5, body_size=8.0):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 3.2, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=title_color)
        ax.text(x + w/2, y + (h - 3.2)/2, content, ha='center', va='center',
                fontsize=body_size, color='#334155', linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, color="#2563eb", lw=1.6, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.5, label, ha='center', va='bottom',
                    fontsize=7.5, color=color, fontweight='bold')

    # ----------------------------------------------------
    # Row 1: Academic Literature Pillars (3 Primary Streams)
    # ----------------------------------------------------
    p1_content = (
        "1. McCulloch & Pitts (1943): Threshold Neuron\n"
        "2. Rosenblatt (1958): Perceptron Learning\n"
        "3. Widrow & Hoff (1960): Delta Rule / Adaline\n"
        "4. Hebb (1949) & Oja (1982): Unsupervised PCA\n"
        "5. Hopfield (1982): Associative Memory\n"
        "6. Rumelhart et al. (1986): Backpropagation"
    )
    draw_box(2, 64, 30, 29, "Pillar 1: Neural Fundamentals (1943-1986)", p1_content, "#f8fafc", "#64748b", "#334155")

    p2_content = (
        "1. Glorot & Bengio (2010): Xavier Initialization\n"
        "2. He et al. (2015): Kaiming Normal (MSRA)\n"
        "3. Srivastava et al. (2014): Dropout Regularization\n"
        "4. Ioffe & Szegedy (2015): Batch Normalization\n"
        "5. Kingma & Ba (2014): Adam Optimizer\n"
        "6. He et al. (2016): Deep Residual Learning"
    )
    draw_box(35, 64, 30, 29, "Pillar 2: Deep Vision Mechanics (2010-2016)", p2_content, "#eff6ff", "#3b82f6", "#1e40af")

    p3_content = (
        "1. Song & Yan (2013): NEU Defect Database\n"
        "   - Canonical 1,800 hot-rolled steel strip images\n"
        "2. Hu et al. (2018): Squeeze-and-Excitation Nets\n"
        "   - Channel attention feature recalibration\n"
        "3. Bao et al. (2021): Steel Defect Detection Survey\n"
        "   - Industrial real-time latency thresholds (<=50ms)"
    )
    draw_box(68, 64, 30, 29, "Pillar 3: Industrial Inspection (2013-2021)", p3_content, "#fefce8", "#eab308", "#854d0e")

    # Connecting arrows to Synthesis Core
    draw_arrow(17, 64, 30, 48, "#475569")
    draw_arrow(50, 64, 50, 48, "#2563eb")
    draw_arrow(83, 64, 70, 48, "#ca8a04")

    # ----------------------------------------------------
    # Row 2: Phase I Platform Implementation Core
    # ----------------------------------------------------
    core_content = (
        "Industrial Surface Defect Inspection Platform (Phase I Core System)\n"
        "- From-Scratch Algorithms: GEMM, SVD Rank Compression, Sigmoid, MLP (XOR/Circle), 4 Optimizers\n"
        "- Leakage-Safe Data Pipeline: Stratified 70/15/15 splits, zero-leakage mu/sigma normalization\n"
        "- BaselineDefectCNN: 28,358 params, 94.44% Acc, 6.95 ms latency (144 FPS edge-ready)\n"
        "- Transfer Learning Benchmark: ResNet-18 ImageNet weights, 99.63% Acc, 40.28 ms latency\n"
        "- Innovation: SE-BaselineDefectCNN with r=4 channel attention (31,102 params, +9.7% overhead)"
    )
    draw_box(10, 31, 80, 17, "Phase I Platform Implementation Architecture", core_content, "#f0fdf4", "#22c55e", "#15803d", title_size=11, body_size=8.5)

    # ----------------------------------------------------
    # Row 3: Open-Source Software Stack & Contribution Roles
    # ----------------------------------------------------
    draw_arrow(28, 31, 28, 24, "#16a34a")
    draw_arrow(72, 31, 72, 24, "#16a34a")

    stack_content = (
        "- Language: Python 3.11.9 (64-bit Windows)\n"
        "- Deep Learning Engine: PyTorch 2.14.1+cpu\n"
        "- Vision Processing: Torchvision 0.29.1+cpu, PIL 10.4\n"
        "- Numerical Engine: NumPy 2.x (BLAS/LAPACK GEMM)\n"
        "- Data Visualization: Matplotlib 3.9 (Publication DPI)\n"
        "- Source Control: Git 2.45 & GitHub Remote"
    )
    draw_box(2, 3, 46, 21, "Open-Source Infrastructure & Libraries", stack_content, "#f1f5f9", "#94a3b8", "#334155")

    roles_content = (
        "- Primary Author / Engineer: Ratna Khanna\n"
        "  * Project conceptualization, architectural design\n"
        "  * Research hypothesis formulation & evaluation criteria\n"
        "  * Experimental validation, viva defense synthesis\n"
        "- AI Pair Programmer: Google DeepMind Antigravity\n"
        "  * Code scaffolding, numerical verification routines\n"
        "  * Script execution, figure generation, and commit logging"
    )
    draw_box(52, 3, 46, 21, "Contribution & Responsibility Allocation", roles_content, "#faf5ff", "#c084fc", "#7e22ce")

    output_dir = PROJECT_ROOT / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "contribution_literature_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved contribution & literature network diagram to {output_path}")

if __name__ == "__main__":
    generate_contribution_network()
