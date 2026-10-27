"""
Industrial Surface Defect Inspection Platform - Phase I
Step 28: Master Viva Voce Preparation Concept Map
Generates a 6-domain comprehensive viva preparation concept map covering all Phase I competencies.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_viva_concept_map():
    fig = plt.figure(figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plt.suptitle("Master Viva Voce Defense Architecture: Industrial Surface Defect Inspection (Phase I)",
                 fontsize=15, fontweight='bold', color='#0f172a', y=0.98)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_domain(x, y, w, h, title, subtitle, bullets, fc, ec, title_color='#0f172a'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                      linewidth=1.6, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.8, title, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=title_color)
        ax.text(x + w/2, y + h - 5.5, subtitle, ha='center', va='center',
                fontsize=8.0, color='#475569', style='italic')
        ax.text(x + w/2, y + (h - 7.5)/2, bullets, ha='center', va='center',
                fontsize=8.0, color='#1e293b', linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, color="#2563eb", lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    # Domain 1: Industrial Problem & Data Governance
    d1_bullets = (
        "- Rolling Line Constraints: 15-20 m/s strip speed\n"
        "- Real-Time Budget: <= 50 ms latency (>= 75 FPS)\n"
        "- NEU Defect Dataset: 1,800 images across 6 classes\n"
        "- Stratified 70/15/15 Split: 1,260 / 270 / 270 (seed=42)\n"
        "- Zero Leakage: Patches_101/105 duplicate in train\n"
        "- Training mu/sigma normalization locked frozen"
    )
    draw_domain(2, 53, 30, 42, "Domain 1: Problem & Data Governance", "Steps 1 to 7", d1_bullets, "#f8fafc", "#64748b", "#0f172a")

    # Domain 2: From-Scratch Mathematics & Neurons
    d2_bullets = (
        "- NumPy GEMM & SVD Rank Compression (99.7% energy)\n"
        "- McCulloch-Pitts: Dual-sensor logic (AND/OR/NOT)\n"
        "- Sigmoid Neuron: Analytical BCE & continuous derivative\n"
        "- Scratch 2-Layer MLP: Multivariate chain rule backprop\n"
        "- Numerical Gradient Checking: Error < 1e-10 (finite diff)\n"
        "- 4 Optimizers: Momentum & NAG (3 ep) vs SGD (19 ep)\n"
        "- 5 Classical Rules: Hebbian, Oja PCA, WTA, Hopfield"
    )
    draw_domain(35, 53, 30, 42, "Domain 2: From-Scratch Mathematics", "Steps 8 to 15", d2_bullets, "#eff6ff", "#3b82f6", "#1d4ed8")

    # Domain 3: Baseline CNN & Optimization
    d3_bullets = (
        "- BaselineDefectCNN: 28,358 params (355.7 KB)\n"
        "- 3 Conv-BN-ReLU stages (16->32->64) + GAP + Dense Head\n"
        "- 20-Epoch Adam Training: Best Val Loss 0.2013 (Ep 18)\n"
        "- Test Accuracy: 94.44% (255/270 correct)\n"
        "- Macro F1: 0.9448 | Recall: 0.9444 (Passed >= 85% SLA)\n"
        "- CPU Latency: 6.95 ms (143.8 FPS - Passed <= 50ms SLA)\n"
        "- Learning Curves: Negative gap proves aug + drop"
    )
    draw_domain(68, 53, 30, 42, "Domain 3: Baseline CNN & Optimization", "Steps 16 to 19", d3_bullets, "#f0fdf4", "#22c55e", "#15803d")

    # Domain 4: Controlled Ablations
    d4_bullets = (
        "- 5-Condition Controlled Study (6 epochs each on CPU):\n"
        "  1. Control (Kaiming + Reg + Aug): 87.78% acc\n"
        "  2. Random Init (sigma=0.02): 78.15% acc (FAILED)\n"
        "  3. Xavier Normal Init: 88.52% acc (stable variance)\n"
        "  4. No Regularization: 86.30% acc (unconstrained fit)\n"
        "  5. No Augmentation: 92.96% acc (memorized stationarity)\n"
        "- Proved Kaiming/Xavier prevent gradient collapse"
    )
    draw_domain(2, 6, 30, 42, "Domain 4: Controlled Ablations", "Step 20", d4_bullets, "#fefce8", "#eab308", "#854d0e")

    # Domain 5: Transfer Learning Benchmark
    d5_bullets = (
        "- Pretrained ResNet-18 (ImageNet-1K, 11.2M params)\n"
        "- 2-Phase Protocol: Ep 1-3 Warmup + Ep 4-8 Fine-Tuning\n"
        "- Test Accuracy: 99.63% (269/270 correct, +5.19% gain)\n"
        "- Macro F1: 0.9963 | CPU Latency: 40.28 ms (24.8 FPS)\n"
        "- Solved pit vs inclusion ambiguity (Inclusion F1 +9.3%)\n"
        "- Error Analysis: Exactly 1 error (intermittent scratch)\n"
        "- Pareto Trade-off: 394x params, 5.8x slower vs Baseline"
    )
    draw_domain(35, 6, 30, 42, "Domain 5: Transfer Learning Benchmark", "Steps 21 & 22", d5_bullets, "#fff7ed", "#f97316", "#c2410c")

    # Domain 6: Innovation Hypothesis & Deployment
    d6_bullets = (
        "- SE-BaselineDefectCNN: Squeeze-and-Excitation (r=4)\n"
        "- Dynamic channel recalibration: s_c = Sigmoid(W2 ReLU(W1 z))\n"
        "- Parameter Budget: 31,102 params (+2,744 params, +9.7%)\n"
        "- Target F1: >= 96.5% | Target Latency: <= 7.5 ms (140 FPS)\n"
        "- Controlled Experiment Plan: 4 questions + McNemar's test\n"
        "- Two-Tier Factory Architecture: Edge Camera (144 FPS)\n"
        "  combined with Central Server Resolver (ResNet-18)"
    )
    draw_domain(68, 6, 30, 42, "Domain 6: Innovation & Deployment", "Steps 23 to 27", d6_bullets, "#faf5ff", "#a855f7", "#7e22ce")

    # Connect Domains
    draw_arrow(32, 74, 35, 74)
    draw_arrow(65, 74, 68, 74)
    draw_arrow(83, 53, 83, 48)
    draw_arrow(68, 27, 65, 27)
    draw_arrow(35, 27, 32, 27)

    output_dir = PROJECT_ROOT / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "viva_preparation_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved viva preparation concept map to {output_path}")

if __name__ == "__main__":
    generate_viva_concept_map()
