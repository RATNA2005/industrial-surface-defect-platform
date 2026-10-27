# Dataset Card: NEU Surface Defect Database (NEU-DET / NEU-CLS)

---

## 1. Dataset Summary & Identification
* **Dataset Name:** NEU Surface Defect Database (`NEU-DET` for detection, `NEU-CLS` for classification)
* **Source:** Key Laboratory of Integrated Automation of Process Industry, Northeastern University (NEU), Shenyang, China
* **Original Authors / Curators:** Prof. Kechen Song and Prof. Yunhui Yan (2013)
* **Source URL:** 
  - Official Faculty Portal: [http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)
  - Canonical Benchmark Mirror: [https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection](https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection)
  - Kaggle Academic Mirror: [https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
* **Licence:** Academic and Educational Research Use Only (Citation required: Song & Yan, 2013)
* **Version:** Canonical Benchmark Edition
* **Access Date:** October 27, 2026
* **Domain:** Industrial Manufacturing Quality Control / Metallurgical Surface Inspection (Hot-Rolled Steel Strip)
* **Purpose:** Academic benchmarking of computer vision and deep learning algorithms for automated defect recognition, localization, and quality assurance in industrial steel strip manufacturing.

---

## 2. Dataset Composition & Statistics
* **Total Images:** 1,800 images
* **Image Resolution:** $200 \times 200$ pixels (100% uniform across all 1,800 images)
* **Image Format & Color Mode:** JPEG (`.jpg`), 3-channel RGB representation of grayscale surface texture
* **Defect Classes (6 Classes):**
  1. **Crazing (`crazing`):** Network of fine micro-cracks on the surface.
  2. **Inclusion (`inclusion`):** Non-metallic metallurgical impurities embedded in the steel.
  3. **Patches (`patches`):** Localized surface oxidation or crust areas.
  4. **Pitted Surface (`pitted_surface`):** Localized micro-cavities caused by chemical etching or mechanical indentation.
  5. **Rolled-in Scale (`rolled-in_scale`):** Iron oxide mill scale pressed into the sheet during hot-rolling.
  6. **Scratches (`scratches`):** Linear abrasions caused by mechanical guide friction or roll slippage.
* **Class Distribution (Image Level):** Perfectly balanced:
  - Crazing: 300 images (16.67%)
  - Inclusion: 300 images (16.67%)
  - Patches: 300 images (16.67%)
  - Pitted Surface: 300 images (16.67%)
  - Rolled-in Scale: 300 images (16.67%)
  - Scratches: 300 images (16.67%)
* **Annotation Type:**
  - Image-level class target via directory / filename designation.
  - Object-level Pascal VOC XML annotations containing 4,189 defect bounding boxes across all 1,800 images (average 2.33 boxes/image).

---

## 3. Data Hygiene & Integrity
* **Cleaning Performed:** 
  - Complete automated file integrity check verified 0 corrupted image files and 0 missing XML pairs.
  - No lossy data trimming or arbitrary removal performed to preserve authentic benchmark integrity.
* **Duplicate Handling:**
  - Automated MD5 hashing detected **1 exact duplicate pair:** `patches_101.jpg` and `patches_105.jpg` (identical pixel values, but independent human bounding box annotations).
  - **Mitigation Strategy:** Both images will be bound together in the training partition (or one sample deduplicated) to guarantee zero crossover into validation/test splits.
  - Perceptual difference hashing (`dHash`) identified 7 near-duplicate clusters from adjacent video-strip frames; these will be tracked to maintain split isolation.
* **Leakage Checks:**
  - Hash-based cross-split verification will be enforced before training.
  - Test set will remain untouched during baseline model tuning and hyperparameter optimization.
  - Preprocessing statistics (e.g. normalization mean and standard deviation) will be computed exclusively on the training split.

---

## 4. Pipeline Strategy (Phase I Implementation)
* **Train / Validation / Test Strategy:**
  - **70% Training Set:** 1,260 images (210 per class)
  - **15% Validation Set:** 270 images (45 per class)
  - **15% Test Set:** 270 images (45 per class)
  - Stratified sampling with a fixed global random seed (`seed=42`).
* **Preprocessing:**
  - Pixel value rescaling to $[0, 1]$ floating-point tensors.
  - Normalization using training split channel statistics ($\mu, \sigma$).
  - One-hot / integer categorical label encoding.
* **Data Augmentation (Training Split Only):**
  - Random horizontal and vertical flips (physically valid on rolling strips).
  - Minor rotations ($\pm 15^\circ$).
  - Minor brightness and contrast adjustments ($\pm 10\%$).
  - **Validation and Test data receive zero random augmentation** (deterministic evaluation only).

---

## 5. Limitations & Contextual Considerations
* **Known Technical Limitations:**
  1. $200 \times 200$ resolution represents localized image crops rather than wide-field mega-pixel line-scan captures.
  2. The dataset focuses purely on defective surfaces; normal defect-free steel exists solely as background pixels outside annotated bounding boxes.
  3. Captured under relatively stable lab lighting, which may not capture extreme rolling mill disturbances (such as dense steam or heavy coolant spray).
* **Potential Biases:**
  - Domain is strictly limited to hot-rolled steel strip; results should not be claimed to generalize to cold-rolled steel, aluminum, or polymers without domain adaptation.
  - Bounding box annotations reflect human operator manual tracing with minor coordinate variance across identical samples (e.g., `patches_101` vs `patches_105`).
* **Ethical Considerations:**
  - Zero PII, human subjects, or biometric attributes are involved.
  - Safety Consideration: The system is designed strictly as human-in-the-loop decision support (PASS / REVIEW / REJECT) to assist QA inspectors, avoiding unverified autonomous line shutdowns or uninspected shipments.
* **Licensing Considerations:**
  - Usable strictly for academic, educational, and non-commercial research under citation of the original paper (Song & Yan, 2013). Commercial deployment requires separate commercial licensing or proprietary industrial data collection.
