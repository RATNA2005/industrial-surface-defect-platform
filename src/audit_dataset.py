import os
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(image, hash_size=8):
    # Grayscale difference hash for near-duplicate detection
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def audit_neu_dataset(images_dir, annotations_dir, figures_dir, report_path):
    print("Starting Comprehensive Dataset Audit...")
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_path.glob('*.*')))
    annotation_files = sorted(list(annotations_path.glob('*.xml')))

    print(f"Discovered {len(image_files)} image files.")
    print(f"Discovered {len(annotation_files)} annotation XML files.")

    audit_results = {
        "total_images": len(image_files),
        "total_annotations": len(annotation_files),
        "image_formats": Counter(),
        "image_dimensions": Counter(),
        "image_modes": Counter(),
        "corrupted_images": [],
        "missing_annotations": [],
        "missing_images": [],
        "classes_by_filename": Counter(),
        "classes_by_xml": Counter(),
        "bounding_boxes_per_class": Counter(),
        "exact_duplicates": [],
        "near_duplicates": [],
        "xml_syntax_errors": [],
        "label_discrepancies": []
    }

    md5_to_files = defaultdict(list)
    dhash_to_files = defaultdict(list)
    image_metadata = {}

    # Audit Images
    for img_file in image_files:
        ext = img_file.suffix.lower()
        audit_results["image_formats"][ext] += 1
        
        # Check corresponding XML
        expected_xml = annotations_path / f"{img_file.stem}.xml"
        if not expected_xml.exists():
            audit_results["missing_annotations"].append(img_file.name)

        try:
            with Image.open(img_file) as img:
                img.verify() # verify integrity
            
            # Reopen to read attributes after verify
            with Image.open(img_file) as img:
                w, h = img.size
                mode = img.mode
                audit_results["image_dimensions"][f"{w}x{h}"] += 1
                audit_results["image_modes"][mode] += 1

                # Infer class from filename prefix
                stem = img_file.stem
                # e.g., 'crazing_12' or 'rolled-in_scale_45'
                prefix = "_".join(stem.split('_')[:-1]) if '_' in stem else stem
                audit_results["classes_by_filename"][prefix] += 1

                # MD5 hash
                md5 = compute_md5(img_file)
                md5_to_files[md5].append(img_file.name)

                # dHash
                dh = compute_dhash(img)
                dhash_to_files[dh].append(img_file.name)

                image_metadata[img_file.stem] = {
                    "filename": img_file.name,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "class_filename": prefix
                }

        except Exception as e:
            audit_results["corrupted_images"].append({"file": img_file.name, "error": str(e)})

    # Audit XML Annotations
    for xml_file in annotation_files:
        stem = xml_file.stem
        # Check corresponding image
        expected_img_exists = any((images_path / f"{stem}{ext}").exists() for ext in ['.bmp', '.jpg', '.jpeg', '.png'])
        if not expected_img_exists:
            audit_results["missing_images"].append(xml_file.name)

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            objects = root.findall('object')
            if not objects:
                audit_results["annotation_problems"].append({"xml": xml_file.name, "issue": "No <object> tags found"})
            
            xml_classes_in_file = set()
            for obj in objects:
                cls_name = obj.find('name').text.strip().lower()
                audit_results["bounding_boxes_per_class"][cls_name] += 1
                xml_classes_in_file.add(cls_name)

                # Check bounding box bounds
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                    audit_results["annotation_problems"].append({
                        "xml": xml_file.name,
                        "issue": f"Invalid bounding box: [{xmin}, {ymin}, {xmax}, {ymax}]"
                    })

            for cls_name in xml_classes_in_file:
                audit_results["classes_by_xml"][cls_name] += 1

            # Check consistency with filename class
            if stem in image_metadata:
                fn_class = image_metadata[stem]["class_filename"].lower()
                # Normalize names (e.g. rolled-in_scale vs rolled-in scale)
                fn_class_norm = fn_class.replace('-', '_').replace(' ', '_')
                xml_classes_norm = [c.replace('-', '_').replace(' ', '_') for c in xml_classes_in_file]
                if fn_class_norm not in xml_classes_norm:
                    audit_results["label_discrepancies"].append({
                        "file": stem,
                        "filename_class": fn_class,
                        "xml_classes": list(xml_classes_in_file)
                    })

        except ET.ParseError as pe:
            audit_results["xml_syntax_errors"].append({"file": xml_file.name, "error": str(pe)})

    # Duplicates Analysis
    for md5, files in md5_to_files.items():
        if len(files) > 1:
            audit_results["exact_duplicates"].append(files)

    for dh, files in dhash_to_files.items():
        if len(files) > 1:
            audit_results["near_duplicates"].append(files)

    # Class Imbalance Analysis
    class_counts = audit_results["classes_by_filename"]
    total = sum(class_counts.values())
    imbalance_ratios = {k: f"{v}/{total} ({v/total*100:.2f}%)" for k, v in class_counts.items()}

    # Convert Counters to dict for JSON serialization
    serializable_report = {
        "total_images": audit_results["total_images"],
        "total_annotations": audit_results["total_annotations"],
        "image_formats": dict(audit_results["image_formats"]),
        "image_dimensions": dict(audit_results["image_dimensions"]),
        "image_modes": dict(audit_results["image_modes"]),
        "corrupted_images": audit_results["corrupted_images"],
        "missing_annotations": audit_results["missing_annotations"],
        "missing_images": audit_results["missing_images"],
        "classes_by_filename": dict(audit_results["classes_by_filename"]),
        "classes_by_xml": dict(audit_results["classes_by_xml"]),
        "bounding_boxes_per_class": dict(audit_results["bounding_boxes_per_class"]),
        "exact_duplicates_count": len(audit_results["exact_duplicates"]),
        "exact_duplicates": audit_results["exact_duplicates"],
        "near_duplicates_groups": len(audit_results["near_duplicates"]),
        "xml_syntax_errors": audit_results["xml_syntax_errors"],
        "label_discrepancies": audit_results["label_discrepancies"],
        "class_proportions": imbalance_ratios
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_report, f, indent=2)
    print(f"Saved audit report to {report_path}")

    # Visualization 1: Class Distribution Bar Chart
    plt.figure(figsize=(9, 5))
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    bars = plt.bar(classes, counts, color='#2b5c8f', edgecolor='black', alpha=0.85)
    plt.title('NEU Surface Defect Database: Class Sample Distribution', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Defect Class', fontsize=11, fontweight='semibold')
    plt.ylabel('Number of Images', fontsize=11, fontweight='semibold')
    plt.ylim(0, max(counts) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 5, f'{h}\n({h/total*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = figures_path / "class_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved class distribution chart to {chart_path}")

    # Visualization 2: Representative Images Grid with Bounding Boxes
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    rep_samples = {}
    for c in classes:
        # Find first sample of this class
        match = [f for f in image_files if f.stem.startswith(c) or f.stem.startswith(c.replace('_', '-'))]
        if match:
            rep_samples[c] = match[0]

    for idx, (cls_name, img_file) in enumerate(rep_samples.items()):
        ax = axes[idx]
        with Image.open(img_file) as img:
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Class: {cls_name}\n({img_file.name})", fontsize=11, fontweight='bold')
            ax.axis('off')

            # Parse XML for bounding box if available
            xml_file = annotations_path / f"{img_file.stem}.xml"
            if xml_file.exists():
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bnd = obj.find('bndbox')
                    xmin = int(bnd.find('xmin').text)
                    ymin = int(bnd.find('ymin').text)
                    xmax = int(bnd.find('xmax').text)
                    ymax = int(bnd.find('ymax').text)
                    rect = patches.Rectangle(
                        (xmin, ymin), xmax - xmin, ymax - ymin,
                        linewidth=2, edgecolor='#e63946', facecolor='none'
                    )
                    ax.add_patch(rect)

    plt.suptitle("NEU Surface Defect Database: Representative Defect Samples & Bounding Boxes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    grid_path = figures_path / "representative_samples.png"
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"Saved representative samples visual to {grid_path}")

    return serializable_report

if __name__ == "__main__":
    audit_neu_dataset(
        images_dir="data/raw/images",
        annotations_dir="data/raw/annotations",
        figures_dir="docs/figures",
        report_path="docs/dataset_audit_report.json"
    )
