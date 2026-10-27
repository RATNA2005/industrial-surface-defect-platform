"""
Baseline Defect CNN Evaluation Script
======================================
Comprehensive evaluation of the trained BaselineDefectCNN on the held-out
test partition (data/splits/test.csv, 270 images).

Outputs:
  - Confusion Matrix (6x6)
  - Per-class Precision, Recall, F1-Score, Support
  - Macro and Weighted Aggregate Metrics
  - Latency and Throughput Benchmarking (CPU)
  - Detailed Misclassification Analysis
  - JSON Metrics Log: docs/baseline_test_metrics.json
  - Visualizations:
      * docs/figures/confusion_matrix_baseline.png
      * docs/figures/misclassified_samples.png
      * docs/figures/confidence_distribution.png
"""

import sys
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import get_dataloaders, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
from src.baseline_cnn import BaselineDefectCNN

def compute_metrics(y_true, y_pred, class_names):
    """
    Computes confusion matrix, per-class precision/recall/F1,
    and macro/weighted averages without relying on external packages.
    """
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
        tn = total_samples - (tp + fp + fn)
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "class_id": i,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "support": int(support),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1, 4))
        }

    overall_accuracy = float(round(np.trace(cm) / total_samples, 4))

    # Macro averages (unweighted arithmetic mean across classes)
    macro_precision = float(round(np.mean([m["precision"] for m in per_class.values()]), 4))
    macro_recall = float(round(np.mean([m["recall"] for m in per_class.values()]), 4))
    macro_f1 = float(round(np.mean([m["f1_score"] for m in per_class.values()]), 4))

    # Weighted averages (weighted by class support)
    weighted_precision = float(round(sum(m["precision"] * m["support"] for m in per_class.values()) / total_samples, 4))
    weighted_recall = float(round(sum(m["recall"] * m["support"] for m in per_class.values()) / total_samples, 4))
    weighted_f1 = float(round(sum(m["f1_score"] * m["support"] for m in per_class.values()) / total_samples, 4))

    return {
        "confusion_matrix": cm.tolist(),
        "overall_accuracy": overall_accuracy,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1
        }
    }

def benchmark_latency(model, device, sample_tensor, iterations=100):
    """
    Measures CPU single-sample inference latency and throughput.
    """
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_tensor)

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(sample_tensor)
            latencies.append((time.perf_counter() - t0) * 1000.0) # in ms

    latencies = np.array(latencies)
    return {
        "mean_latency_ms": float(round(np.mean(latencies), 2)),
        "median_latency_ms": float(round(np.median(latencies), 2)),
        "p95_latency_ms": float(round(np.percentile(latencies, 95), 2)),
        "min_latency_ms": float(round(np.min(latencies), 2)),
        "max_latency_ms": float(round(np.max(latencies), 2)),
        "std_latency_ms": float(round(np.std(latencies), 2)),
        "throughput_fps": float(round(1000.0 / np.mean(latencies), 1))
    }

def plot_confusion_matrix(cm_list, class_names, out_path):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="BaselineDefectCNN: Test Set Confusion Matrix\n(Held-out Test Partition, N=270)",
        ylabel="True Defect Class",
        xlabel="Predicted Defect Class"
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Annotate counts and row-wise recall percentage
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_total = cm[i, :].sum()
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_total * 100.0) if row_total > 0 else 0.0
            text_str = f"{val}\n({pct:.1f}%)" if val > 0 else "0"
            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9, fontweight="bold" if i == j else "normal"
            )

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to: {out_path}")

def plot_misclassified_samples(misclassified, out_path, max_samples=12):
    if not misclassified:
        print("No misclassified samples found to plot.")
        return

    n_samples = min(len(misclassified), max_samples)
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_samples):
        sample = misclassified[idx]
        img_path = sample["image_path"]
        with Image.open(img_path) as img:
            axes[idx].imshow(img)

        title_text = (
            f"{sample['filename']}\n"
            f"True: {sample['true_class']}\n"
            f"Pred: {sample['pred_class']} ({sample['confidence']*100:.1f}%)"
        )
        axes[idx].set_title(title_text, fontsize=9, color="darkred", fontweight="bold", pad=8)
        axes[idx].axis('off')

        # Add red border around misclassified sample
        for spine in axes[idx].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Turn off remaining empty subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.35, wspace=0.25)
    plt.suptitle(
        f"BaselineDefectCNN: Representative Misclassifications on Held-out Test Set\n(Total Errors: {len(misclassified)} / 270)",
        fontsize=12, fontweight="bold", y=0.98
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved misclassified samples figure to: {out_path}")

def plot_confidence_distribution(correct_confidences, error_confidences, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)

    ax.hist(
        correct_confidences, bins=bins, alpha=0.7, color="#2a9d8f",
        label=f"Correct Predictions (N={len(correct_confidences)}, Mean={np.mean(correct_confidences):.2f})",
        edgecolor="black"
    )
    if len(error_confidences) > 0:
        ax.hist(
            error_confidences, bins=bins, alpha=0.8, color="#e63946",
            label=f"Misclassifications (N={len(error_confidences)}, Mean={np.mean(error_confidences):.2f})",
            edgecolor="black"
        )

    ax.set_title("Prediction Confidence Distribution (Softmax Probabilities)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class Probability (Confidence)", fontsize=10)
    ax.set_ylabel("Number of Test Samples", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confidence distribution figure to: {out_path}")

def evaluate_baseline_model(
    checkpoint_path="models/baseline_cnn_best.pth",
    splits_dir="data/splits",
    output_json="docs/baseline_test_metrics.json",
    figures_dir="docs/figures"
):
    print("=" * 75)
    print("  PHASE I: BASELINE DEFECT CNN TEST EVALUATION (STEP 18)")
    print("=" * 75)

    device = torch.device("cpu")
    print(f"Execution Device:       {device}")

    # Load Model
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(ckpt_file, map_location=device)
    print(f"Loaded Checkpoint from: {checkpoint_path}")
    print(f"  Best Val Epoch:       {checkpoint.get('epoch')}")
    print(f"  Best Val Loss:        {checkpoint.get('val_loss'):.5f}")
    print(f"  Best Val Accuracy:    {checkpoint.get('val_acc'):.2f}%")

    model = BaselineDefectCNN(in_channels=3, num_classes=6, dropout_rate=0.25).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load DataLoaders
    _, _, test_loader, mean, std = get_dataloaders(splits_dir=splits_dir, batch_size=32, seed=42)
    total_test = len(test_loader.dataset)
    print(f"Test Partition Size:    {total_test} samples across 6 classes")

    # Run Benchmark Latency on CPU
    dummy_sample = torch.randn(1, 3, 200, 200).to(device)
    latency_stats = benchmark_latency(model, device, dummy_sample, iterations=100)
    print(f"Single-Image Latency:   Mean={latency_stats['mean_latency_ms']} ms | P95={latency_stats['p95_latency_ms']} ms | FPS={latency_stats['throughput_fps']}")

    # Evaluation loop
    y_true = []
    y_pred = []
    all_confidences = []
    correct_confidences = []
    error_confidences = []
    misclassified_records = []
    detailed_records = []

    # Map filename to image_path from test_dataset.samples
    file_to_path = {s["filename"]: s["image_path"] for s in test_loader.dataset.samples}

    with torch.no_grad():
        for images, labels, class_names_batch, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(images.size(0)):
                t_label = labels[i].item()
                p_label = preds[i].item()
                conf = confs[i].item()
                fname = filenames[i]
                t_name = CLASS_NAMES[t_label]
                p_name = CLASS_NAMES[p_label]
                prob_dist = [round(float(p), 4) for p in probs[i].cpu().numpy()]

                y_true.append(t_label)
                y_pred.append(p_label)
                all_confidences.append(conf)

                rec = {
                    "filename": fname,
                    "image_path": str(file_to_path[fname]).replace("\\", "/"),
                    "true_class": t_name,
                    "true_id": t_label,
                    "pred_class": p_name,
                    "pred_id": p_label,
                    "confidence": round(conf, 4),
                    "probabilities": prob_dist,
                    "correct": (t_label == p_label)
                }
                detailed_records.append(rec)

                if t_label == p_label:
                    correct_confidences.append(conf)
                else:
                    error_confidences.append(conf)
                    misclassified_records.append(rec)

    # Sort misclassifications by confidence descending (most confident mistakes first)
    misclassified_records.sort(key=lambda x: x["confidence"], reverse=True)

    # Compute all classification metrics
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)

    # Assemble full result dictionary
    results = {
        "evaluation_summary": {
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "test_samples": total_test,
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_precision": metrics["macro_avg"]["precision"],
            "macro_recall": metrics["macro_avg"]["recall"],
            "macro_f1": metrics["macro_avg"]["f1_score"],
            "weighted_precision": metrics["weighted_avg"]["precision"],
            "weighted_recall": metrics["weighted_avg"]["recall"],
            "weighted_f1": metrics["weighted_avg"]["f1_score"],
            "total_errors": len(misclassified_records),
            "error_rate": round(len(misclassified_records) / total_test, 4)
        },
        "latency_benchmark_cpu": latency_stats,
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": {
            "class_names": CLASS_NAMES,
            "matrix": metrics["confusion_matrix"]
        },
        "misclassified_samples": misclassified_records
    }

    # Print Formatted Report to Console
    print("\n" + "=" * 75)
    print("  TEST EVALUATION METRIC REPORT")
    print("=" * 75)
    print(f"  Overall Accuracy:    {metrics['overall_accuracy'] * 100:.2f}% ({total_test - len(misclassified_records)} / {total_test} correct)")
    print(f"  Macro Precision:     {metrics['macro_avg']['precision']:.4f}")
    print(f"  Macro Recall:        {metrics['macro_avg']['recall']:.4f}")
    print(f"  Macro F1-Score:      {metrics['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted F1-Score:   {metrics['weighted_avg']['f1_score']:.4f}")
    print(f"  Total Misclassified: {len(misclassified_records)} / {total_test} ({len(misclassified_records)/total_test*100:.2f}%)")
    print("-" * 75)
    print(f"{'Class Name':<18} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Support':<7}")
    print("-" * 75)
    for cname in CLASS_NAMES:
        m = metrics["per_class"][cname]
        print(f"{cname:<18} | {m['precision']:<9.4f} | {m['recall']:<9.4f} | {m['f1_score']:<9.4f} | {m['support']:<7d}")
    print("-" * 75)
    print(f"{'Macro Average':<18} | {metrics['macro_avg']['precision']:<9.4f} | {metrics['macro_avg']['recall']:<9.4f} | {metrics['macro_avg']['f1_score']:<9.4f} | {total_test:<7d}")
    print(f"{'Weighted Average':<18} | {metrics['weighted_avg']['precision']:<9.4f} | {metrics['weighted_avg']['recall']:<9.4f} | {metrics['weighted_avg']['f1_score']:<9.4f} | {total_test:<7d}")
    print("=" * 75)

    print(f"\nTop Misclassified Samples (Most Confident Errors):")
    for idx, err in enumerate(misclassified_records[:6], 1):
        print(f"  [{idx}] {err['filename']}: True={err['true_class']:<15} -> Pred={err['pred_class']:<15} (Conf={err['confidence']*100:.1f}%)")

    # Generate Visualizations
    fig_path = Path(figures_dir)
    cm_fig_path = fig_path / "confusion_matrix_baseline.png"
    misc_fig_path = fig_path / "misclassified_samples.png"
    conf_fig_path = fig_path / "confidence_distribution.png"

    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_fig_path)
    plot_misclassified_samples(misclassified_records, misc_fig_path, max_samples=12)
    plot_confidence_distribution(correct_confidences, error_confidences, conf_fig_path)

    # Save JSON log
    out_json_path = Path(output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics JSON to: {out_json_path}")

    return results

if __name__ == "__main__":
    evaluate_baseline_model()
