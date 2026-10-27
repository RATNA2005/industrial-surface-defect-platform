"""
Industrial Surface Defect Inspection Platform - Phase I
Step 24: Controlled Experiment Plan Visualization
Generates a structured flowchart visualizing the 4-question experimental protocol for evaluating the Innovation Hypothesis.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_experiment_flowchart():
    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plt.suptitle("Controlled Experiment Protocol: Testing the SE-BaselineDefectCNN Innovation",
                 fontsize=15, fontweight='bold', color='#111827', y=0.96)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_box(x, y, w, h, title, content, fc, ec, title_color='#111827'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                      linewidth=1.8, edgecolor=ec, facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 3.5, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=title_color)
        ax.text(x + w/2, y + (h - 3.5)/2, content, ha='center', va='center',
                fontsize=8.5, color='#334155', linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#1e40af"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 2, label, ha='center', va='bottom',
                    fontsize=8.5, color="#1e3a8a", fontweight='bold')

    # Column 1: What Changed?
    q1_content = (
        "- Insert 3 Squeeze-and-Excitation units\n"
        "- Reduction ratio r=4 (16->4, 32->8, 64->16)\n"
        "- Global Average Pooling squeeze\n"
        "- Two-layer MLP bottleneck with Sigmoid\n"
        "- +2,744 parameters (+9.7% over 28,358)\n"
        "- Target model: SE-BaselineDefectCNN"
    )
    draw_box(3, 52, 21, 38, "1. What Changed?", q1_content, "#eff6ff", "#3b82f6", "#1d4ed8")

    # Column 2: Why?
    q2_content = (
        "- Baseline CNN had 15 errors on test set\n"
        "- Inclusions F1: 89.6% (8 false positives)\n"
        "- Pitted Surface F1: 92.0% (5 false negatives)\n"
        "- Hard to separate from roller striations\n"
        "- SE dynamically amplifies defect channels\n"
        "- Suppresses background strip textures"
    )
    draw_box(27, 52, 21, 38, "2. Why?", q2_content, "#fefce8", "#eab308", "#a16207")

    # Column 3: What Happened? (Controls)
    q3_content = (
        "- Strict Control: BaselineDefectCNN\n"
        "- Exact same split: 1,260 train / 270 test\n"
        "- Seed: 42 (zero data leakage)\n"
        "- Same Adam optimizer (lr=1e-3, wd=1e-4)\n"
        "- Same 20-epoch budget & augmentations\n"
        "- Baseline Reference: 94.44% Acc, 0.9448 F1"
    )
    draw_box(51, 52, 21, 38, "3. What Happened?", q3_content, "#f0fdf4", "#22c55e", "#15803d")

    # Column 4: Did It Improve? (Decision Rules)
    q4_content = (
        "- Primary SLA: Macro F1 >= 96.5% (+2.0%)\n"
        "- Difficult Classes: Inclusions >= 93.5%\n"
        "- Pitted Surfaces F1 >= 95.5%\n"
        "- Efficiency: Parameters <= 31,500\n"
        "- Latency: CPU <= 8.0 ms (>= 125 FPS)\n"
        "- Reject H0 if all 5 SLA criteria pass"
    )
    draw_box(75, 52, 22, 38, "4. Did It Improve?", q4_content, "#faf5ff", "#a855f7", "#7e22ce")

    # Horizontal Flow Arrows between the 4 Questions
    draw_arrow(24, 71, 27, 71)
    draw_arrow(48, 71, 51, 71)
    draw_arrow(72, 71, 75, 71)

    # Bottom Protocol Execution Stages
    draw_box(3, 8, 28, 34, "Phase A: Implementation & Training",
             "1. Define SEBlock & SEBaselineDefectCNN in PyTorch\n"
             "2. Initialize weights via Kaiming normal + Xavier\n"
             "3. Train for 20 epochs using train.csv & val.csv\n"
             "4. Save best checkpoint to models/se_baseline_best.pth",
             "#f8fafc", "#94a3b8")

    draw_box(35, 8, 29, 34, "Phase B: Evaluation & Benchmark",
             "1. Evaluate on isolated test.csv (N=270)\n"
             "2. Benchmark CPU inference latency (B=1, 270 passes)\n"
             "3. Compute Confusion Matrix & per-class F1-scores\n"
             "4. Log metrics to docs/se_baseline_metrics.json",
             "#f8fafc", "#94a3b8")

    draw_box(68, 8, 29, 34, "Phase C: Hypothesis Decision Gate",
             "1. Compute delta against Baseline (Step 18)\n"
             "2. Conduct McNemar's Test for significance (p < 0.05)\n"
             "3. Check Edge Hardware SLA: Latency <= 8.0 ms\n"
             "4. If PASSED: Validate H1 & proceed to Phase II roadmap",
             "#f8fafc", "#94a3b8")

    # Downward connecting arrows
    draw_arrow(13.5, 52, 17, 42)
    draw_arrow(49.5, 42, 49.5, 42)
    draw_arrow(31, 25, 35, 25)
    draw_arrow(64, 25, 68, 25)

    output_dir = PROJECT_ROOT / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_plan_flowchart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved controlled experiment flowchart to {output_path}")

if __name__ == "__main__":
    generate_experiment_flowchart()
