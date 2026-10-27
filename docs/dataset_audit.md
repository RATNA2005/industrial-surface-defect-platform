# Comprehensive Dataset Audit Report: NEU Surface Defect Database

**Audit Execution Date:** October 27, 2026  
**Audited Location:** `data/raw/images` and `data/raw/annotations`  
**Automated Audit Script:** [`src/audit_dataset.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/audit_dataset.py)  
**Machine-Readable Audit Record:** [`docs/dataset_audit_report.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/dataset_audit_report.json)

---

## 1. Summary Metrics

| Attribute | Audited Value | Finding & Integrity Status |
| :--- | :---: | :--- |
| **Total Images Discovered** | **1,800** | All expected images present |
| **Total Annotations Discovered** | **1,800** | Exactly 1:1 mapping with images |
| **File Formats** | **100% `.jpg`** | 1,800 JPEG images |
| **Image Dimensions** | **$200 \times 200$** | 100% uniform resolution |
| **Color Mode** | **RGB (3 channels)** | Grayscale intensity replicated across 3 channels |
| **Corrupted Image Files** | **0** | Verified openable and decodable via PIL |
| **Missing XML / Image Pairs** | **0** | 100% paired pairing |
| **XML Syntax / Parse Errors** | **0** | All 1,800 XML files parsed cleanly |
| **Class Imbalance (Image Level)**| **0%** | Exactly 300 images per class ($16.67\%$) |

---

## 2. Class Distribution & Bounding Box Census

| Defect Class | Filename Prefix | Total Images | Image % | Bounding Box Count | Mean Boxes / Image |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Crazing** | `crazing` | 300 | 16.67% | 689 | 2.30 |
| **Inclusion** | `inclusion` | 300 | 16.67% | 1,011 | 3.37 |
| **Patches** | `patches` | 300 | 16.67% | 881 | 2.94 |
| **Pitted Surface** | `pitted_surface` | 300 | 16.67% | 432 | 1.44 |
| **Rolled-in Scale** | `rolled-in_scale` | 300 | 16.67% | 628 | 2.09 |
| **Scratches** | `scratches` | 300 | 16.67% | 548 | 1.83 |
| **Total** | — | **1,800** | **100.0%** | **4,189** | **2.33** |

---

## 3. Critical Anomaly & Leakage Analysis

### 3.1 Exact Duplicate Discovery
* **Identified Duplicate Pair:** `patches_101.jpg` and `patches_105.jpg`
* **Evidence:** Identical MD5 hash (`b93228...`).
* **Annotation Nuance:** While the pixels are 100% identical, the XML annotations were manually drawn separately:
  - `patches_101.xml`: Box 1 `[27, 1, 78, 92]`, Box 2 `[113, 1, 175, 59]`, Box 3 `[129, 79, 176, 200]`
  - `patches_105.xml`: Box 1 `[29, 1, 76, 92]`, Box 2 `[116, 1, 174, 53]`, Box 3 `[130, 78, 173, 200]`
* **Action Required:** In Step 6, both samples must either be placed together in the training set or one sample must be deduplicated to prevent train-test data leakage.

### 3.2 Near-Duplicate Clusters
* Perceptual difference hashing (`dHash`) revealed 7 clusters of high structural similarity, primarily among faint micro-inclusions and low-contrast pitted surfaces.
* **Cause:** Surface defects captured along adjacent line-scan camera frames during continuous rolling.

### 3.3 Normal vs. Defective Ratio
* All 1,800 raw images depict localized defective surfaces.
* Defect-free surface regions exist extensively as background within each $200 \times 200$ patch outside the designated bounding boxes.

---

## 4. Visual Evidence Artifacts
* Class distribution bar chart generated: [`docs/figures/class_distribution.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/class_distribution.png)
* Representative sample grid with ground-truth bounding boxes generated: [`docs/figures/representative_samples.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/representative_samples.png)
