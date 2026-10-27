# Leakage-Safe Stratified Data Splitting Protocol

**Execution Date:** October 27, 2026  
**Script:** [`src/create_splits.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/create_splits.py)  
**Machine-Readable Summary:** [`data/splits/split_summary.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/data/splits/split_summary.json)  
**Fixed Random Seed:** `42`

---

## 1. Partition Ratios & Sample Counts

| Partition | Total Images | Percentage | Per-Class Quota (6 Classes) | Role in Phase I |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | **1,260** | **70.0%** | **210 images / class** | Used for model weight updates and loss backpropagation |
| **Validation Set** | **270** | **15.0%** | **45 images / class** | Used for hyperparameter tuning, model checkpointing & early stopping |
| **Test Set** | **270** | **15.0%** | **45 images / class** | **Held-out strictly untouched** until final unbiased performance reporting |
| **Total** | **1,800** | **100.0%** | **300 images / class** | Complete dataset census |

---

## 2. Per-Class Stratification Breakdown

| Defect Class | Training (70%) | Validation (15%) | Test (15%) | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Crazing (`Cr`)** | 210 | 45 | 45 | 300 |
| **Inclusion (`In`)** | 210 | 45 | 45 | 300 |
| **Patches (`Pa`)** | 210 | 45 | 45 | 300 |
| **Pitted Surface (`PS`)** | 210 | 45 | 45 | 300 |
| **Rolled-in Scale (`RS`)** | 210 | 45 | 45 | 300 |
| **Scratches (`Sc`)** | 210 | 45 | 45 | 300 |
| **Overall Sum** | **1,260** | **270** | **270** | **1,800** |

---

## 3. Leakage Mitigation & Duplicate Binding Evidence

Automated integrity verification confirmed all 5 leakage safeguards:
1. **Filename Disjointness:** 
   - $\text{Train} \cap \text{Val} = \emptyset$ (0 shared filenames)
   - $\text{Train} \cap \text{Test} = \emptyset$ (0 shared filenames)
   - $\text{Val} \cap \text{Test} = \emptyset$ (0 shared filenames)
2. **Hash Collisions (MD5 Verification):**
   - 0 identical byte-level hashes exist across different partition boundaries.
3. **Exact Duplicate Group Assignment:**
   - Both `patches_101.jpg` and `patches_105.jpg` (which share identical image pixels) were explicitly bound into the **Training Partition**. Neither appears in the Validation or Test partitions.
4. **Independent Test Partition:**
   - The test partition is saved to [`data/splits/test.csv`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/data/splits/test.csv) and will remain untouched during training loop development.
5. **Preprocessing Isolation:**
   - Normalization statistics (mean, std) will be computed exclusively on `data/splits/train.csv` and applied to `val.csv` and `test.csv` without recomputing.

---

## 4. Visual Distribution
- Visual representation of partition quotas: [`docs/figures/split_distribution.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/split_distribution.png)
