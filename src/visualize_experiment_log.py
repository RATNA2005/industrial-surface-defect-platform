"""
Industrial Surface Defect Inspection Platform - Phase I
Step 25: Master Experiment Progression Visualization
Generates a multi-panel publication dashboard summarizing all Phase I model runs, ablations, and benchmarks.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_experiment_summary_plot():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plt.suptitle("Phase I Master Experiment Progression: Steel Surface Defect Inspection",
                 fontsize=15, fontweight='bold', color='#111827', y=0.98)

    # ----------------------------------------------------
    # Subplot 1: Test Accuracy across Experimental Runs
    # ----------------------------------------------------
    models = [
        "MLP Scratch\n(XOR/Circle)",
        "Ablation 2\n(Random Init)",
        "Ablation 4\n(No Reg)",
        "Ablation 1\n(Control 6ep)",
        "Ablation 3\n(Xavier Init)",
        "Ablation 5\n(No Aug)",
        "Baseline CNN\n(20 epochs)",
        "ResNet-18\n(Pretrained)"
    ]
    accuracies = [100.0, 78.15, 86.30, 87.78, 88.52, 92.96, 94.44, 99.63]
    colors = ['#94a3b8', '#ef4444', '#f59e0b', '#3b82f6', '#06b6d4', '#8b5cf6', '#0d9488', '#ea580c']

    bars1 = ax1.bar(models, accuracies, color=colors, width=0.6, edgecolor='#1e293b', linewidth=1.2)
    ax1.set_ylim(70, 105)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight='bold')
    ax1.set_title("(A) Accuracy Across Phase I Experimental Runs", fontsize=12, fontweight='bold', loc='left')
    ax1.axhline(85.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Industrial SLA Threshold (85%)')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9)

    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{acc:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.setp(ax1.get_xticklabels(), rotation=25, ha='right', fontsize=8.5)

    # ----------------------------------------------------
    # Subplot 2: Parameter Footprint (Log Scale)
    # ----------------------------------------------------
    param_models = [
        "MLP (Scratch)",
        "Baseline CNN\n(Scratch)",
        "SE-Baseline\n(Innovation)",
        "ResNet-18\n(Transfer)"
    ]
    param_counts = [17, 28358, 31102, 11179590]
    param_colors = ['#64748b', '#0d9488', '#10b981', '#ea580c']

    bars2 = ax2.bar(param_models, param_counts, color=param_colors, width=0.55, edgecolor='#1e293b', linewidth=1.2)
    ax2.set_yscale('log')
    ax2.set_ylabel("Parameter Count (Logarithmic)", fontsize=11, fontweight='bold')
    ax2.set_title("(B) Parameter Footprint Scaling Across Architectural Classes", fontsize=12, fontweight='bold', loc='left')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    for bar, count in zip(bars2, param_counts):
        if count < 1000:
            label = f"{count}"
        elif count < 100000:
            label = f"{count/1000:.1f} K"
        else:
            label = f"{count/1e6:.2f} M"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.35,
                 label, ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # ----------------------------------------------------
    # Subplot 3: CPU Inference Latency & Line Speed SLA
    # ----------------------------------------------------
    latency_models = [
        "MLP Scratch\n(Synthetic)",
        "Baseline CNN\n(Test Partition)",
        "SE-Baseline\n(Estimated)",
        "ResNet-18\n(Test Partition)"
    ]
    latencies = [0.08, 6.95, 7.15, 40.28]
    lat_colors = ['#64748b', '#0d9488', '#10b981', '#ea580c']

    bars3 = ax3.bar(latency_models, latencies, color=lat_colors, width=0.55, edgecolor='#1e293b', linewidth=1.2)
    ax3.set_ylabel("CPU Mean Latency per Frame (ms)", fontsize=11, fontweight='bold')
    ax3.set_title("(C) CPU Latency vs. Manufacturing Real-Time SLA", fontsize=12, fontweight='bold', loc='left')
    ax3.axhline(50.0, color='#dc2626', linestyle='--', linewidth=1.8, label='Maximum Real-Time SLA Limit (50 ms)')
    ax3.set_ylim(0, 58)
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=9)

    for bar, lat in zip(bars3, latencies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f"{lat:.2f} ms\n({1000/lat:.0f} FPS)", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # ----------------------------------------------------
    # Subplot 4: Latency vs. Test Accuracy Pareto Frontier
    # ----------------------------------------------------
    ax4.set_title("(D) Industrial Pareto Frontier: Speed vs. Accuracy", fontsize=12, fontweight='bold', loc='left')
    ax4.set_xlabel("Mean Inference Latency (ms) [Lower is Better]", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Test Accuracy (%) [Higher is Better]", fontsize=11, fontweight='bold')

    # Plot points
    ax4.scatter([6.95], [94.44], s=350, color='#0d9488', edgecolors='#0f766e', linewidth=2.0, zorder=5, label='BaselineDefectCNN (28K params)')
    ax4.scatter([7.15], [97.00], s=380, color='#10b981', edgecolors='#047857', linewidth=2.0, zorder=5, marker='*', label='SE-BaselineDefectCNN (Projected Target)')
    ax4.scatter([40.28], [99.63], s=650, color='#ea580c', edgecolors='#c2410c', linewidth=2.0, zorder=5, label='Pretrained ResNet-18 (11.2M params)')

    # Annotations
    ax4.annotate("BaselineDefectCNN\n94.4% Acc | 7.0 ms\n(144 FPS Edge Ready)",
                 xy=(6.95, 94.44), xytext=(12, 92.5),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="#0f766e"),
                 fontsize=8.5, fontweight='bold', color="#0f766e")

    ax4.annotate("SE-Baseline (Target)\n>=96.5% Acc | 7.2 ms\n(Attention Recalibrated)",
                 xy=(7.15, 97.00), xytext=(12, 96.2),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="#047857"),
                 fontsize=8.5, fontweight='bold', color="#047857")

    ax4.annotate("Pretrained ResNet-18\n99.6% Acc | 40.3 ms\n(25 FPS Server Grade)",
                 xy=(40.28, 99.63), xytext=(24, 98.6),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="#c2410c"),
                 fontsize=8.5, fontweight='bold', color="#c2410c")

    ax4.axvline(50.0, color='#dc2626', linestyle='--', linewidth=1.5)
    ax4.text(50.5, 91.0, "50 ms SLA Threshold", color='#dc2626', fontsize=8.5, fontweight='bold', rotation=90)

    ax4.set_xlim(0, 55)
    ax4.set_ylim(90, 101)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    output_dir = PROJECT_ROOT / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_progression_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved experiment progression summary plot to {output_path}")

if __name__ == "__main__":
    generate_experiment_summary_plot()
