import csv
import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def create_leakage_safe_splits(
    images_dir="data/raw/images",
    annotations_dir="data/raw/annotations",
    splits_dir="data/splits",
    figures_dir="docs/figures",
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
):
    print("Executing Leakage-Safe Stratified Splitting...")
    random.seed(seed)
    np.random.seed(seed)

    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    splits_path = Path(splits_dir)
    figures_path = Path(figures_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_path.glob('*.jpg')))
    assert len(image_files) == 1800, f"Expected 1800 images, found {len(image_files)}"

    # Class definitions (ordered alphabetically)
    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]
    class_to_id = {cls: idx for idx, cls in enumerate(class_names)}

    # Group images by class and compute MD5
    class_to_items = defaultdict(list)
    md5_to_items = defaultdict(list)

    for img_path in image_files:
        stem = img_path.stem
        # Extract class prefix
        cls_name = "_".join(stem.split('_')[:-1]) if '_' in stem else stem
        assert cls_name in class_to_id, f"Unknown class {cls_name} from {img_path.name}"
        
        xml_path = annotations_path / f"{stem}.xml"
        assert xml_path.exists(), f"Missing XML for {img_path.name}"
        
        md5 = compute_md5(img_path)
        item = {
            "filename": img_path.name,
            "stem": stem,
            "image_path": str(img_path).replace("\\", "/"),
            "annotation_path": str(xml_path).replace("\\", "/"),
            "class_name": cls_name,
            "class_id": class_to_id[cls_name],
            "md5": md5
        }
        class_to_items[cls_name].append(item)
        md5_to_items[md5].append(item)

    # Verify known exact duplicates (patches_101 and patches_105)
    duplicate_groups = [items for items in md5_to_items.values() if len(items) > 1]
    print(f"Identified {len(duplicate_groups)} exact duplicate group(s):")
    for grp in duplicate_groups:
        filenames = [x["filename"] for x in grp]
        print(f"  -> MD5 Duplicate Group: {filenames}")

    train_set = []
    val_set = []
    test_set = []

    # Stratified split per class with explicit duplicate binding
    for cls_name in class_names:
        items = class_to_items[cls_name]
        assert len(items) == 300, f"Class {cls_name} has {len(items)} items, expected 300"

        # Separate items that belong to multi-item MD5 groups so they are assigned as an indivisible unit
        unique_items = []
        dup_items = []

        for itm in items:
            if len(md5_to_items[itm["md5"]]) > 1:
                dup_items.append(itm)
            else:
                unique_items.append(itm)

        # Shuffle unique items deterministically
        random.shuffle(unique_items)

        # Calculate quotas: 210 train, 45 val, 45 test per class
        target_train = int(round(300 * train_ratio)) # 210
        target_val = int(round(300 * val_ratio))     # 45
        target_test = 300 - target_train - target_val # 45

        # All exact duplicates are bound to the TRAINING partition to eliminate risk of test leakage
        cls_train = list(dup_items)
        needed_train = target_train - len(cls_train)
        
        cls_train.extend(unique_items[:needed_train])
        remaining = unique_items[needed_train:]

        cls_val = remaining[:target_val]
        cls_test = remaining[target_val:target_val + target_test]

        assert len(cls_train) == target_train, f"Train count mismatch: {len(cls_train)} != {target_train}"
        assert len(cls_val) == target_val, f"Val count mismatch: {len(cls_val)} != {target_val}"
        assert len(cls_test) == target_test, f"Test count mismatch: {len(cls_test)} != {target_test}"

        train_set.extend(cls_train)
        val_set.extend(cls_val)
        test_set.extend(cls_test)

    # Verification: Total sizes
    assert len(train_set) == 1260, f"Train set size {len(train_set)} != 1260"
    assert len(val_set) == 270, f"Val set size {len(val_set)} != 270"
    assert len(test_set) == 270, f"Test set size {len(test_set)} != 270"
    assert len(train_set) + len(val_set) + len(test_set) == 1800

    # Verification: Filename overlap check (must be mutually disjoint)
    train_files = {x["filename"] for x in train_set}
    val_files = {x["filename"] for x in val_set}
    test_files = {x["filename"] for x in test_set}

    assert len(train_files & val_files) == 0, f"Overlap between train and val: {train_files & val_files}"
    assert len(train_files & test_files) == 0, f"Overlap between train and test: {train_files & test_files}"
    assert len(val_files & test_files) == 0, f"Overlap between val and test: {val_files & test_files}"

    # Verification: MD5 hash leakage check
    train_hashes = {x["md5"] for x in train_set}
    val_hashes = {x["md5"] for x in val_set}
    test_hashes = {x["md5"] for x in test_set}

    md5_train_val = train_hashes & val_hashes
    md5_train_test = train_hashes & test_hashes
    md5_val_test = val_hashes & test_hashes

    assert len(md5_train_val) == 0, f"MD5 Leakage between train and val: {md5_train_val}"
    assert len(md5_train_test) == 0, f"MD5 Leakage between train and test: {md5_train_test}"
    assert len(md5_val_test) == 0, f"MD5 Leakage between val and test: {md5_val_test}"

    print("ALL LEAKAGE INTEGRITY CHECKS PASSED:")
    print("  [PASSED] Zero filename overlap across splits.")
    print("  [PASSED] Zero MD5 hash collisions across splits.")
    print("  [PASSED] Exact duplicate pair ('patches_101.jpg', 'patches_105.jpg') safely bound inside train split.")
    print("  [PASSED] Exactly 70% Train (1260), 15% Val (270), 15% Test (270).")

    # Save CSV Splits
    def write_split_csv(split_data, out_path):
        fieldnames = ["filename", "stem", "image_path", "annotation_path", "class_name", "class_id", "md5"]
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(split_data, key=lambda x: x["filename"]):
                writer.writerow(row)
        print(f"Saved: {out_path} ({len(split_data)} rows)")

    write_split_csv(train_set, splits_path / "train.csv")
    write_split_csv(val_set, splits_path / "val.csv")
    write_split_csv(test_set, splits_path / "test.csv")

    # Generate Class-wise Census per Split
    def count_classes(dataset):
        c = defaultdict(int)
        for item in dataset:
            c[item["class_name"]] += 1
        return c

    train_counts = count_classes(train_set)
    val_counts = count_classes(val_set)
    test_counts = count_classes(test_set)

    summary_data = {
        "dataset_name": "NEU Surface Defect Database",
        "random_seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_images": 1800,
        "split_sizes": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set)
        },
        "class_breakdown": {
            cls: {
                "train": train_counts[cls],
                "val": val_counts[cls],
                "test": test_counts[cls],
                "total": train_counts[cls] + val_counts[cls] + test_counts[cls]
            } for cls in class_names
        },
        "leakage_checks": {
            "filename_overlap_train_val": len(train_files & val_files),
            "filename_overlap_train_test": len(train_files & test_files),
            "filename_overlap_val_test": len(val_files & test_files),
            "md5_collision_train_val": len(md5_train_val),
            "md5_collision_train_test": len(md5_train_test),
            "md5_collision_val_test": len(md5_val_test),
            "duplicate_pair_assignment": {
                "patches_101.jpg": "train" if "patches_101.jpg" in train_files else "other",
                "patches_105.jpg": "train" if "patches_105.jpg" in train_files else "other"
            }
        }
    }

    summary_file = splits_path / "split_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved: {summary_file}")

    # Visualization: Split Distribution Stacked Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(class_names))
    width = 0.55

    tr_vals = [train_counts[c] for c in class_names]
    va_vals = [val_counts[c] for c in class_names]
    te_vals = [test_counts[c] for c in class_names]

    p1 = plt.bar(x, tr_vals, width, label='Train (70% = 210)', color='#1d3557', edgecolor='black')
    p2 = plt.bar(x, va_vals, width, bottom=tr_vals, label='Val (15% = 45)', color='#457b9d', edgecolor='black')
    p3 = plt.bar(x, te_vals, width, bottom=np.array(tr_vals)+np.array(va_vals), label='Test (15% = 45)', color='#e63946', edgecolor='black')

    plt.ylabel('Image Count', fontsize=11, fontweight='bold')
    plt.title('Leakage-Safe Stratified Split Distribution (70% Train / 15% Val / 15% Test)', fontsize=12, fontweight='bold', pad=15)
    plt.xticks(x, class_names, rotation=15, fontweight='semibold')
    plt.ylim(0, 350)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(class_names)):
        plt.text(i, 305, f"Total: {tr_vals[i]+va_vals[i]+te_vals[i]}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = figures_path / "split_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved split distribution chart to: {chart_path}")

    return summary_data

if __name__ == "__main__":
    create_leakage_safe_splits()
