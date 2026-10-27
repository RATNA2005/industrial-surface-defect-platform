# Industrial Surface Defect Inspection Platform

[![Deep Learning Hackathon](https://img.shields.io/badge/Project-Deep%20Learning%20Hackathon-blue.svg)](https://github.com/RATNA2005/industrial-surface-defect-platform)
[![Phase](https://img.shields.io/badge/Status-Phase%20I%20In%20Progress-green.svg)](https://github.com/RATNA2005/industrial-surface-defect-platform)

An automated image-based visual quality inspection system for industrial manufacturing, focusing on **hot-rolled steel strip surface defect classification and inspection**.

The platform is designed to provide automated decision support for Quality Assurance (QA) inspectors by classifying surface defect patches and categorizing items into **PASS**, **REVIEW**, or **REJECT** workflows.

---

## 📌 Project Overview
- **Domain:** Manufacturing Quality / Metal Surface Defect Inspection
- **Current Target:** Phase I (Foundations, Data Pipeline, Fundamentals, Baseline CNN)
- **Approved Benchmark Dataset:** NEU Surface Defect Database (1,800 images across 6 balanced defect classes)
- **Decision Architecture:** Human-in-the-loop decision-support pipeline

---

## 📂 Repository Structure
```
industrial-surface-defect-platform/
├── data/                      # Dataset documentation and raw storage
│   └── README.md
├── docs/                      # Formal academic documentation and reports
│   ├── dataset_provenance.md  # Step 3: Source, license, authors, and custodial trail
│   ├── dataset_audit.md       # Step 4: Empirical audit report and integrity checks
│   ├── dataset_audit_report.json # Machine-readable audit statistics
│   └── figures/               # Audit visualizations (distribution, samples)
│       ├── class_distribution.png
│       └── representative_samples.png
├── src/                       # Source code modules
│   └── audit_dataset.py       # Automated dataset audit script
├── requirements.txt           # Project dependencies
├── .gitignore                 # Version control hygiene
└── README.md                  # Project overview
```

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/RATNA2005/industrial-surface-defect-platform.git
cd industrial-surface-defect-platform
pip install -r requirements.txt
```

### 2. Run the Dataset Audit
To re-run the automated dataset audit, verify 1,800 images, and regenerate distribution charts:
```bash
python src/audit_dataset.py
```

---

## 📊 Completed Milestones (Phase I)
- [x] **Step 1:** Industrial Use Case Definition (`PASS / REVIEW / REJECT` workflow)
- [x] **Step 2:** Research & Proposal of Candidate Datasets (NEU-DET, GC10-DET, Severstal, MVTec AD)
- [x] **Step 3:** Dataset Approval & Provenance Logging (`docs/dataset_provenance.md`)
- [x] **Step 4:** Comprehensive Empirical Dataset Audit (`docs/dataset_audit.md`)
- [ ] **Step 5:** Dataset Card Creation
- [ ] **Step 6:** Leakage-Safe Data Splitting (70% / 15% / 15%)
- [ ] **Step 7:** Reproducible Data Loading & Preprocessing Pipeline
- [ ] **Steps 8–15:** Deep Learning Mathematical Fundamentals (Tensors, Neurons, MLP from Scratch)
- [ ] **Steps 16–20:** Baseline CNN Classifier Implementation & Experiments
- [ ] **Steps 21–28:** Advanced Architecture & Phase I Viva Preparation
