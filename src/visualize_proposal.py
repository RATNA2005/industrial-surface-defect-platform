"""
Industrial Surface Defect Inspection Platform - Phase I
Step 27: Executive Proposal Infographic Visualization
Generates an executive one-page visual banner synthesizing the problem, benchmarks, innovation, and deployment architecture.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_proposal_infographic():
    fig = plt.figure(figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plt.suptitle("Executive One-Page Summary: Industrial Surface Defect Inspection Platform (Phase I)",
                 fontsize=15, fontweight='bold', color='#0f172a', y=0.98)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_card(x, y, w, h, title, content, fc, ec, title_color='#0f172a', body_color='#334155'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 3.2, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=title_color)
        ax.text(x + w/2, y + (h - 3.2)/2, content, ha='center', va='center',
                fontsize=8.2, color=body_color, linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, color="#2563eb", lw=1.6):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    # ----------------------------------------------------
    # Header Banner: Problem & Operational SLA
    # ----------------------------------------------------
    header_content = (
        "DOMAIN: Hot-Rolled Steel Strip Manufacturing (Rolling Line Speeds: 15 - 20 m/s)\n"
        "PROBLEM: Optical defect leakage causes automotive structural flaws and mill line stoppages.\n"
        "ACCEPTANCE CRITERIA: Macro F1 >= 85.0%, Defect Recall >= 90.0%, CPU Latency <= 50.0 ms (Throughput >= 75 FPS)\n"
        "CANONICAL DATASET: NEU Surface Defect Database (1,800 images, 6 classes: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches)"
    )
    draw_card(2, 80, 96, 15, "1. Industrial Problem Statement & Operational Constraints",
              header_content, "#f8fafc", "#64748b", "#1e293b", "#0f172a")

    # ----------------------------------------------------
    # Middle Row: Mathematical Foundations | Baseline CNN | Transfer ResNet-18
    # ----------------------------------------------------
    math_content = (
        "- Pure NumPy implementations\n"
        "- SVD low-rank compression (99.7% energy)\n"
        "- McCulloch-Pitts inspection logic\n"
        "- Sigmoid & multivariate MLP backpropagation\n"
        "- Numerical gradient checking (error < 1e-10)\n"
        "- 4 Optimizers (NAG, Momentum, SGD, GD)\n"
        "- 5 Learning Rules (Oja PCA, Hopfield Memory)"
    )
    draw_card(2, 43, 30, 34, "2. Foundations from Scratch", math_content, "#eff6ff", "#3b82f6", "#1d4ed8")

    baseline_content = (
        "- Model: BaselineDefectCNN (Custom)\n"
        "- Parameters: 28,358 (355.7 KB checkpoint)\n"
        "- Test Accuracy: 94.44% (255/270 correct)\n"
        "- Macro F1-Score: 0.9448 (Recall: 94.44%)\n"
        "- CPU Latency: 6.95 ms (143.8 FPS)\n"
        "- SLA Compliance: PASSED (7.2x faster than SLA)\n"
        "- Limitation: 10/15 errors on pits vs inclusions"
    )
    draw_card(35, 43, 30, 34, "3. Lightweight Baseline CNN", baseline_content, "#f0fdf4", "#22c55e", "#15803d")

    resnet_content = (
        "- Model: Pretrained ResNet-18 (ImageNet-1K)\n"
        "- Parameters: 11,179,590 (42.72 MB checkpoint)\n"
        "- Test Accuracy: 99.63% (269/270 correct)\n"
        "- Macro F1-Score: 0.9963 (Recall: 99.63%)\n"
        "- CPU Latency: 40.28 ms (24.8 FPS)\n"
        "- Delta: +5.19% Acc, +0.0515 F1 over Baseline\n"
        "- Trade-Off: 394x more params, 5.8x slower"
    )
    draw_card(68, 43, 30, 34, "4. Transfer Learning Benchmark", resnet_content, "#fff7ed", "#f97316", "#c2410c")

    # Connect middle cards
    draw_arrow(32, 60, 35, 60)
    draw_arrow(65, 60, 68, 60)

    # ----------------------------------------------------
    # Bottom Row: Innovation Hypothesis | Two-Tier Deployment | Phase II Roadmap
    # ----------------------------------------------------
    innov_content = (
        "SE-BaselineDefectCNN Architecture:\n"
        "- Squeeze-and-Excitation channel attention (r=4)\n"
        "- Dynamically recalibrates micro-texture channels\n"
        "- Parameter Budget: 31,102 (+9.7% overhead)\n"
        "- Latency Target: ~7.1 ms (140 FPS on CPU)\n"
        "- F1 Target: >= 96.5% (+2.0% over Baseline)"
    )
    draw_card(2, 6, 30, 34, "5. Innovation: SE Channel Attention", innov_content, "#faf5ff", "#a855f7", "#7e22ce")

    deploy_content = (
        "Two-Tier Factory Inspection System:\n"
        "- Tier 1 (Line Camera DSP): Baseline / SE-CNN\n"
        "  * Operates at 144 FPS directly on camera\n"
        "  * Low-confidence frames flagged for Tier 2\n"
        "- Tier 2 (Central Server): Pretrained ResNet-18\n"
        "  * 99.63% verification of flagged defects\n"
        "  * Actuates pneumatic strip rejector"
    )
    draw_card(35, 6, 30, 34, "6. Two-Tier Deployment System", deploy_content, "#fefce8", "#eab308", "#854d0e")

    roadmap_content = (
        "Phase II Advanced Milestones:\n"
        "1. Localization: YOLOv8 / RetinaNet bboxes\n"
        "2. Segmentation: U-Net pixel-level defect depth\n"
        "3. Generative AI: Diffusion/GAN rare defect synth\n"
        "4. Edge UI: Real-time full-stack inspection app\n"
        "5. ONNX / TensorRT hardware acceleration"
    )
    draw_card(68, 6, 30, 34, "7. Phase II Scope & Roadmap", roadmap_content, "#f1f5f9", "#475569", "#1e293b")

    # Connecting arrows downward
    draw_arrow(50, 43, 50, 40)
    draw_arrow(17, 43, 17, 40)
    draw_arrow(83, 43, 83, 40)

    output_dir = PROJECT_ROOT / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase_I_executive_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved phase I executive summary infographic to {output_path}")

if __name__ == "__main__":
    generate_proposal_infographic()
