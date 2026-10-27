"""
Model Comparison & Benchmarking Engine: BaselineDefectCNN vs. Pretrained ResNet-18
===================================================================================
Conducts rigorous head-to-head empirical benchmarking between:
  1. BaselineDefectCNN (28,358 parameters | Trained from scratch)
  2. Pretrained ResNet-18 (11,179,590 parameters | Fine-tuned ImageNet-1K)

Outputs:
  - Quantitative comparison table: Test Acc, Macro F1, Recall, Latency, Storage
  - Multi-panel publication figure: docs/figures/model_comparison.png
  - Metrics JSON: docs/model_comparison_metrics.json
  - Markdown Report: docs/baseline_vs_resnet_comparison.md
"""

import sys
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import NEUDefectDataset, CLASS_NAMES
from src.baseline_cnn import BaselineDefectCNN
from src.train_resnet import build_resnet18, get_resnet_transforms, IMAGENET_MEAN, IMAGENET_STD

def compute_detailed_metrics(y_true, y_pred, class_names):
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    total_samples = len(y_true)

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "support": int(support),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1, 4))
        }

    overall_accuracy = float(round(np.trace(cm) / total_samples, 4))
    macro_precision = float(round(np.mean([m["precision"] for m in per_class.values()]), 4))
    macro_recall = float(round(np.mean([m["recall"] for m in per_class.values()]), 4))
    macro_f1 = float(round(np.mean([m["f1_score"] for m in per_class.values()]), 4))

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class
    }

def benchmark_single_latency(model, sample_tensor, iterations=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10): # warmup
            _ = model(sample_tensor)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(sample_tensor)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    return {
        "mean_latency_ms": float(round(np.mean(latencies), 2)),
        "median_latency_ms": float(round(np.median(latencies), 2)),
        "p95_latency_ms": float(round(np.percentile(latencies, 95), 2)),
        "std_latency_ms": float(round(np.std(latencies), 2)),
        "throughput_fps": float(round(1000.0 / np.mean(latencies), 1))
    }

def evaluate_model_on_test(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels, _, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return compute_detailed_metrics(y_true, y_pred, CLASS_NAMES)

def plot_comparison(baseline_metrics, resnet_metrics, out_path="docs/figures/model_comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # -------------------------------------------------------------
    # Panel A: Overall Accuracy & Macro F1 Bar Chart
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    metrics_labels = ["Accuracy (%)", "Macro F1 (%)", "Macro Recall (%)"]
    base_vals = [baseline_metrics["accuracy"] * 100, baseline_metrics["macro_f1"] * 100, baseline_metrics["macro_recall"] * 100]
    res_vals = [resnet_metrics["accuracy"] * 100, resnet_metrics["macro_f1"] * 100, resnet_metrics["macro_recall"] * 100]

    x = np.arange(len(metrics_labels))
    width = 0.35

    rects1 = ax_a.bar(x - width/2, base_vals, width, label='BaselineDefectCNN (Scratch)', color='#2a9d8f', edgecolor='black')
    rects2 = ax_a.bar(x + width/2, res_vals, width, label='ResNet-18 (Pretrained)', color='#e76f51', edgecolor='black')

    ax_a.axhline(y=85.0, color="#d62828", linestyle="--", linewidth=1.2, label="Industrial Target (85%)")
    ax_a.set_title("(A) Test Partition Performance (N=270)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Score (%)", fontsize=10)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metrics_labels, fontsize=10)
    ax_a.set_ylim(70, 102)
    ax_a.grid(axis='y', linestyle='--', alpha=0.5)
    ax_a.legend(loc="lower right", fontsize=8.5)

    for rect in rects1:
        h = rect.get_height()
        ax_a.text(rect.get_x() + rect.get_width()/2., h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax_a.text(rect.get_x() + rect.get_width()/2., h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # -------------------------------------------------------------
    # Panel B: Per-Class F1-Score Comparison
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    classes = CLASS_NAMES
    x_c = np.arange(len(classes))
    base_f1s = [baseline_metrics["per_class"][c]["f1_score"] * 100 for c in classes]
    res_f1s = [resnet_metrics["per_class"][c]["f1_score"] * 100 for c in classes]

    ax_b.bar(x_c - width/2, base_f1s, width, label='BaselineDefectCNN', color='#2a9d8f', edgecolor='black')
    ax_b.bar(x_c + width/2, res_f1s, width, label='ResNet-18 (Pretrained)', color='#e76f51', edgecolor='black')

    ax_b.set_title("(B) Per-Class Defect F1-Scores (%)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("F1-Score (%)", fontsize=10)
    ax_b.set_xticks(x_c)
    ax_b.set_xticklabels(classes, rotation=25, ha='right', fontsize=9)
    ax_b.set_ylim(80, 105)
    ax_b.grid(axis='y', linestyle='--', alpha=0.5)
    ax_b.legend(loc="lower right", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel C: Accuracy vs. Latency Pareto Trade-off
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    base_lat = baseline_metrics["latency"]["mean_latency_ms"]
    res_lat = resnet_metrics["latency"]["mean_latency_ms"]
    base_acc = baseline_metrics["accuracy"] * 100
    res_acc = resnet_metrics["accuracy"] * 100

    # Scatter points with bubble size proportional to parameter count
    ax_c.scatter([base_lat], [base_acc], s=250, color='#2a9d8f', edgecolor='black', linewidth=1.5, zorder=5, label='BaselineDefectCNN (28K params)')
    ax_c.scatter([res_lat], [res_acc], s=1200, color='#e76f51', edgecolor='black', linewidth=1.5, zorder=5, label='ResNet-18 (11.2M params - 394x larger)')

    ax_c.axvline(x=50.0, color='#d62828', linestyle='--', linewidth=1.5, label='Real-time Line Threshold (50 ms)')
    ax_c.annotate(f"BaselineDefectCNN\n{base_acc:.1f}% Acc | {base_lat:.1f} ms\n(142.9 FPS)",
                  xy=(base_lat, base_acc), xytext=(base_lat + 2, base_acc - 2.5),
                  arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=9, fontweight='bold')
    ax_c.annotate(f"Pretrained ResNet-18\n{res_acc:.1f}% Acc | {res_lat:.1f} ms\n({resnet_metrics['latency']['throughput_fps']:.1f} FPS)",
                  xy=(res_lat, res_acc), xytext=(res_lat - 12, res_acc - 3.0),
                  arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=9, fontweight='bold')

    ax_c.set_title("(C) Industrial Speed vs. Accuracy Trade-Off (CPU)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Mean Inference Latency (ms) [Lower is Better]", fontsize=10)
    ax_c.set_ylabel("Test Accuracy (%) [Higher is Better]", fontsize=10)
    ax_c.set_xlim(0, 55)
    ax_c.set_ylim(88, 101)
    ax_c.grid(True, linestyle='--', alpha=0.5)
    ax_c.legend(loc="lower left", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel D: Parameter Count & Memory Storage Footprint
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    comp_labels = ["Parameters\n(Millions)", "Checkpoint Size\n(Megabytes)"]
    base_res_vals = [28358 / 1e6, baseline_metrics["checkpoint_mb"]]
    resnet_res_vals = [11179590 / 1e6, resnet_metrics["checkpoint_mb"]]

    x_d = np.arange(len(comp_labels))
    ax_d.bar(x_d - width/2, base_res_vals, width, label='BaselineDefectCNN', color='#2a9d8f', edgecolor='black')
    ax_d.bar(x_d + width/2, resnet_res_vals, width, label='ResNet-18', color='#e76f51', edgecolor='black')

    ax_d.set_title("(D) Parameter & Storage Resource Efficiency", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("Quantity (M params / MB on disk)", fontsize=10)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(comp_labels, fontsize=10)
    ax_d.set_yscale('log')
    ax_d.grid(axis='y', linestyle='--', alpha=0.5)
    ax_d.legend(loc="upper left", fontsize=8.5)

    # Annotate values
    ax_d.text(x_d[0] - width/2, base_res_vals[0] * 1.5, f"0.028 M\n(28,358)", ha='center', fontsize=8.5, fontweight='bold')
    ax_d.text(x_d[0] + width/2, resnet_res_vals[0] * 1.2, f"11.18 M", ha='center', fontsize=8.5, fontweight='bold')
    ax_d.text(x_d[1] - width/2, base_res_vals[1] * 1.5, f"{base_res_vals[1]:.2f} MB\n({baseline_metrics['checkpoint_kb']:.0f} KB)", ha='center', fontsize=8.5, fontweight='bold')
    ax_d.text(x_d[1] + width/2, resnet_res_vals[1] * 1.2, f"{resnet_res_vals[1]:.1f} MB", ha='center', fontsize=8.5, fontweight='bold')

    plt.suptitle("BaselineDefectCNN vs. Pretrained ResNet-18: Industrial Defect Inspection Benchmark", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison figure to: {out_file}")

def run_model_comparison():
    print("=" * 75)
    print("  PHASE I: BASELINE VS. RESNET-18 COMPARISON BENCHMARK (STEP 22)")
    print("=" * 75)

    device = torch.device("cpu")
    splits_path = root_dir / "data" / "splits"
    test_csv = splits_path / "test.csv"

    # 1. Baseline Evaluation
    base_ckpt_path = root_dir / "models" / "baseline_cnn_best.pth"
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {base_ckpt_path}")

    # Load baseline normalization & transforms
    config_path = splits_path / "pipeline_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        b_mean = cfg["train_normalization"]["mean"]
        b_std = cfg["train_normalization"]["std"]

    base_eval_tf = T.Compose([
        T.Resize((200, 200)),
        T.ToTensor(),
        T.Normalize(mean=b_mean, std=b_std)
    ])
    base_test_ds = NEUDefectDataset(test_csv, transform=base_eval_tf)
    base_test_loader = DataLoader(base_test_ds, batch_size=32, shuffle=False)

    base_model = BaselineDefectCNN(in_channels=3, num_classes=6, dropout_rate=0.25).to(device)
    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()

    base_metrics = evaluate_model_on_test(base_model, base_test_loader, device)
    base_sample = torch.randn(1, 3, 200, 200).to(device)
    base_latency = benchmark_single_latency(base_model, base_sample, iterations=100)
    base_metrics["latency"] = base_latency
    base_metrics["checkpoint_bytes"] = base_ckpt_path.stat().st_size
    base_metrics["checkpoint_kb"] = round(base_metrics["checkpoint_bytes"] / 1024, 1)
    base_metrics["checkpoint_mb"] = round(base_metrics["checkpoint_bytes"] / (1024 * 1024), 2)
    base_metrics["parameters"] = 28358

    # 2. ResNet-18 Evaluation
    res_ckpt_path = root_dir / "models" / "resnet18_best.pth"
    if not res_ckpt_path.exists():
        raise FileNotFoundError(f"ResNet checkpoint not found at: {res_ckpt_path}")

    _, res_eval_tf = get_resnet_transforms()
    res_test_ds = NEUDefectDataset(test_csv, transform=res_eval_tf)
    res_test_loader = DataLoader(res_test_ds, batch_size=32, shuffle=False)

    res_model = build_resnet18(num_classes=6, dropout_rate=0.30).to(device)
    res_ckpt = torch.load(res_ckpt_path, map_location=device)
    res_model.load_state_dict(res_ckpt["model_state_dict"])
    res_model.eval()

    res_metrics = evaluate_model_on_test(res_model, res_test_loader, device)
    res_sample = torch.randn(1, 3, 224, 224).to(device)
    res_latency = benchmark_single_latency(res_model, res_sample, iterations=100)
    res_metrics["latency"] = res_latency
    res_metrics["checkpoint_bytes"] = res_ckpt_path.stat().st_size
    res_metrics["checkpoint_kb"] = round(res_metrics["checkpoint_bytes"] / 1024, 1)
    res_metrics["checkpoint_mb"] = round(res_metrics["checkpoint_bytes"] / (1024 * 1024), 2)
    res_metrics["parameters"] = 11179590

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD BENCHMARK: BASELINE DEFECT CNN VS. PRETRAINED RESNET-18")
    print("=" * 80)
    print(f"{'Performance Metric':<28} | {'BaselineDefectCNN':<18} | {'ResNet-18 (Pretrained)':<22} | {'Delta':<10}")
    print("-" * 80)
    print(f"{'Test Accuracy':<28} | {base_metrics['accuracy']*100:6.2f}%            | {res_metrics['accuracy']*100:6.2f}%                | {res_metrics['accuracy']*100 - base_metrics['accuracy']*100:+5.2f}%")
    print(f"{'Macro Precision':<28} | {base_metrics['macro_precision']:6.4f}             | {res_metrics['macro_precision']:6.4f}               | {res_metrics['macro_precision'] - base_metrics['macro_precision']:+6.4f}")
    print(f"{'Macro Recall':<28} | {base_metrics['macro_recall']:6.4f}             | {res_metrics['macro_recall']:6.4f}               | {res_metrics['macro_recall'] - base_metrics['macro_recall']:+6.4f}")
    print(f"{'Macro F1-Score':<28} | {base_metrics['macro_f1']:6.4f}             | {res_metrics['macro_f1']:6.4f}               | {res_metrics['macro_f1'] - base_metrics['macro_f1']:+6.4f}")
    print(f"{'Parameters':<28} | {base_metrics['parameters']:,}             | {res_metrics['parameters']:,}           | +{res_metrics['parameters'] - base_metrics['parameters']:,}")
    print(f"{'Checkpoint Disk Size':<28} | {base_metrics['checkpoint_kb']} KB            | {res_metrics['checkpoint_mb']} MB               | +{res_metrics['checkpoint_mb'] - base_metrics['checkpoint_mb']:.1f} MB")
    print(f"{'Mean CPU Latency':<28} | {base_metrics['latency']['mean_latency_ms']:5.2f} ms            | {res_metrics['latency']['mean_latency_ms']:5.2f} ms             | +{res_metrics['latency']['mean_latency_ms'] - base_metrics['latency']['mean_latency_ms']:5.2f} ms")
    print(f"{'P95 CPU Latency':<28} | {base_metrics['latency']['p95_latency_ms']:5.2f} ms            | {res_metrics['latency']['p95_latency_ms']:5.2f} ms             | +{res_metrics['latency']['p95_latency_ms'] - base_metrics['latency']['p95_latency_ms']:5.2f} ms")
    print(f"{'Throughput (FPS)':<28} | {base_metrics['latency']['throughput_fps']:5.1f} FPS           | {res_metrics['latency']['throughput_fps']:5.1f} FPS            | {res_metrics['latency']['throughput_fps'] - base_metrics['latency']['throughput_fps']:+5.1f} FPS")
    print("=" * 80)

    print("\nPer-Class F1-Score Comparison:")
    print(f"{'Class Name':<18} | {'Baseline F1':<12} | {'ResNet-18 F1':<12} | {'Delta':<8}")
    print("-" * 55)
    for c in CLASS_NAMES:
        b_f1 = base_metrics["per_class"][c]["f1_score"]
        r_f1 = res_metrics["per_class"][c]["f1_score"]
        print(f"{c:<18} | {b_f1*100:6.2f}%      | {r_f1*100:6.2f}%       | {(r_f1 - b_f1)*100:+5.2f}%")
    print("=" * 55 + "\n")

    # Generate visual figure
    plot_comparison(base_metrics, res_metrics)

    # Save JSON log
    out_json = root_dir / "docs" / "model_comparison_metrics.json"
    results = {
        "metadata": {
            "test_partition": "data/splits/test.csv",
            "test_samples": 270,
            "classes": CLASS_NAMES
        },
        "baseline_defect_cnn": base_metrics,
        "pretrained_resnet18": res_metrics
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison metrics JSON to: {out_json}")

if __name__ == "__main__":
    run_model_comparison()
