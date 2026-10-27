# Reproducible Data Pipeline & Preprocessing Protocol

**Implementation Modules:**
- Dataset Class & Loaders: [`src/dataset.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/dataset.py)
- Verification & Test Script: [`src/test_pipeline.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/test_pipeline.py)
- Configuration Specification: [`data/splits/pipeline_config.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/data/splits/pipeline_config.json)

---

## 1. Zero-Leakage Normalization Statistics

Channel statistics were computed **strictly across the 1,260 images of the training partition** (`data/splits/train.csv`). No validation or test image pixels were included.

$$\mu_{\text{train}} = [0.5040, 0.5040, 0.5040], \quad \sigma_{\text{train}} = [0.2121, 0.2121, 0.2121]$$

*(Identical across all 3 RGB channels because raw images represent single-channel grayscale surface textures).*

---

## 2. Transformation Pipelines

```
Training Pipeline (Domain-Appropriate Augmentation)
  Raw Image (200x200)
       ↓
  Resize(200, 200)
       ↓
  RandomHorizontalFlip(p=0.5)      [Physically valid: Steel rolling direction symmetry]
       ↓
  RandomVerticalFlip(p=0.5)        [Physically valid: Invariant strip surface plane]
       ↓
  RandomRotation(±15°)             [Simulates minor camera mounting / guide rail skew]
       ↓
  ColorJitter(b=0.1, c=0.1)        [Simulates minor illumination / coil reflectivity drift]
       ↓
  ToTensor()                       [Scales uint8 [0, 255] to float32 [0.0, 1.0]]
       ↓
  Normalize(μ_train, σ_train)      [Zero-mean, unit-variance standardized tensor]

Evaluation Pipeline (Strictly Deterministic: Validation & Test)
  Raw Image (200x200)
       ↓
  Resize(200, 200)
       ↓
  ToTensor()
       ↓
  Normalize(μ_train, σ_train)
```

> [!IMPORTANT]
> **Deterministic Evaluation Guarantee:**  
> The validation and test pipelines contain **zero random transformations**. Every validation and test image receives the exact same deterministic transformation throughout the project life-cycle.

---

## 3. DataLoader Specifications

| Attribute | Training Loader | Validation Loader | Test Loader |
| :--- | :---: | :---: | :---: |
| **Dataset Manifest** | `data/splits/train.csv` | `data/splits/val.csv` | `data/splits/test.csv` |
| **Total Samples** | 1,260 | 270 | 270 |
| **Batch Size** | 32 | 32 | 32 |
| **Total Batches** | 40 batches (39 full + 1 of 12) | 9 batches (8 full + 1 of 14) | 9 batches (8 full + 1 of 14) |
| **Shuffle Policy** | **True** (Worker seeded with `42 + worker_id`) | **False** (Deterministic order) | **False** (Deterministic order) |
| **Output Image Shape** | `[32, 3, 200, 200]` (`float32`) | `[32, 3, 200, 200]` (`float32`) | `[32, 3, 200, 200]` (`float32`) |
| **Output Label Shape** | `[32]` (`int64`, values 0–5) | `[32]` (`int64`, values 0–5) | `[32]` (`int64`, values 0–5) |

---

## 4. Visual Evidence
- Augmentation demonstration figure: [`docs/figures/pipeline_samples.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/pipeline_samples.png)
