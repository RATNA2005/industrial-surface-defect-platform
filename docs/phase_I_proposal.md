# Phase I Executive Project Proposal: Industrial Surface Defect Inspection Platform

**Document Version:** 1.0.0  
**Phase:** Phase I — Deep Learning Foundations & Model Comparison (Step 27)  
**Lead Author:** Ratna Khanna (`ratnakhanna2005@gmail.com`)  
**Project Title:** Industrial Surface Defect Inspection Platform (Hot-Rolled Steel Strip Surface Defect Inspection)  
**Target Domain:** Manufacturing Quality Assurance & Edge Automated Optical Inspection (AOI)  
**Canonical Dataset:** NEU Surface Defect Database (1,800 images across 6 defect classes)  
**GitHub Repository:** `https://github.com/RATNA2005/industrial-surface-defect-platform.git` (Branch: `main`)  

---

## 1. Executive Summary & Problem Formulation

In modern hot-rolling steel mills, continuous steel strips travel through finishing stands at line speeds exceeding **$15\text{ to }20\text{ m/s}$ ($54\text{ to }72\text{ km/h}$)**. Undetected surface defects—such as non-metallic slag inclusions, pitted surfaces, fatigue crazing cracks, or rolling scratches—can cause catastrophic strip ruptures, unscheduled production shutdowns costing over **$\$50,000\text{ per hour}$**, and severe structural failures in downstream automotive press-forming and pipeline welding applications.

Manual stroboscopic visual inspection catches less than $65\%$ of defects and is physically impossible at line speeds above $5\text{ m/s}$. To achieve **100% contiguous optical inspection**, digital line-scan cameras must capture and evaluate non-overlapping $0.2\text{ m}$ strip segments at a minimum throughput of **$75\text{ to }100\text{ FPS}$**, imposing a rigid **real-time latency budget of $\le 50.0\text{ ms}$ per frame** on embedded edge hardware mounted directly inside harsh, vibration-heavy mill enclosures.

```
+----------------------------------------------------------------------------------------------------+
|                                    PROJECT MISSION STATEMENT                                       |
| Establish a mathematically rigorous, auditable Phase I deep learning foundation for 6-class steel  |
| surface defect inspection: from first-principles neural derivations to an ultra-lightweight        |
| baseline CNN (144 FPS on CPU), transfer learning benchmarking (99.6% accuracy), and an             |
| edge-friendly channel attention innovation (SE-BaselineDefectCNN) for mill deployment.             |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Phase I System Overview & Infographic

The end-to-end Phase I engineering architecture, baseline vs. transfer benchmarks, and proposed deployment framework are synthesized in the executive infographic below:

![Phase I Executive Summary Infographic](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/phase_I_executive_summary.png)

*Figure 1: High-density executive synthesis connecting industrial constraints, from-scratch mathematical foundations, empirical baseline vs. transfer benchmarks, channel attention innovation, and two-tier deployment.*

---

## 3. Mathematical & Algorithmic Foundations (From Scratch)

Before deploying deep neural networks, we validated core computational mechanics from first principles in pure NumPy without automated autograd frameworks:

1. **Tensor Sandbox & GEMM:** Implemented vectorized matrix multiplication and stable softmax. SVD low-rank decomposition proved that rank $k=20$ preserves **$99.7\%$ of variance**, demonstrating that localized defect edges lie on low-dimensional manifolds. Numerical gradient verification confirmed machine precision ($1.06 \times 10^{-11} \ll 10^{-7}$).
2. **Neuron Mechanics:** Derived and trained McCulloch-Pitts threshold neurons (dual-sensor inspection logic: AND, OR, inhibitory AND-NOT), Sigmoid logistic neurons with analytical binary cross-entropy, and a 2-layer MLP (17 parameters) that solved non-linear XOR and circular boundary problems to 100% accuracy.
3. **Optimization Dynamics:** Proved that Mini-batch SGD with Nesterov Accelerated Gradient (NAG) overcomes non-convex saddle points in 3 epochs, whereas standard Batch Gradient Descent stalls indefinitely.
4. **Classical Learning Rules:** Verified Hebbian correlation, Oja’s PCA rule (error $<0.018$), Competitive Winner-Take-All clustering, Widrow-Hoff LMS delta rule, and Hopfield associative memory energy minimization ($E=0 \to -6$) for scratch denoising.

---

## 4. Empirical Benchmarks: Baseline CNN vs. Pretrained ResNet-18

We developed, trained, and benchmarked two contrasting deep learning architectures on the held-out test partition ($N=270$, 45 samples per class across 6 defect types) under identical single-threaded CPU execution:

| Evaluation Dimension | Industrial SLA Target | BaselineDefectCNN (From Scratch) | Pretrained ResNet-18 (Transfer Learning) | Operational Trade-Off / Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Model Parameters** | Lightweight Edge Friendly | **28,358** | **11,179,590** | Baseline is **$394.2\times$ smaller** |
| **Disk Checkpoint Size** | $< 100\text{ MB}$ | **355.7 KB** ($0.35\text{ MB}$) | **42.72 MB** ($43,747\text{ KB}$) | Baseline fits into L2/L3 cache |
| **Test Accuracy** | $\ge 85.0\%$ | **94.44%** (255/270) | **99.63%** (269/270) | ResNet-18 is **$+5.19\%$ superior** |
| **Macro F1-Score** | $\ge 85.0\%$ | **0.9448** (94.48%) | **0.9963** (99.63%) | ResNet-18 achieves **$+0.0515$ gain** |
| **Macro Recall** | $\ge 90.0\%$ | **0.9444** (94.44%) | **0.9963** (99.63%) | Both pass safety-critical recall SLA |
| **Inclusion F1-Score** | Difficult Class | 0.8958 (89.58%) | **0.9890 (98.90%)** | ResNet resolves subtle inclusions ($+9.32\%$) |
| **Pitted Surface F1** | Difficult Class | 0.9195 (91.95%) | **1.0000 (100.0%)** | ResNet resolves shallow craters ($+8.05\%$) |
| **CPU Mean Latency ($B=1$)** | $\le 50.0\text{ ms}$ | **6.95 ms** | **40.28 ms** | Baseline is **$5.8\times$ faster** |
| **Throughput (FPS on CPU)** | $\ge 75\text{ FPS}$ ($15\text{ m/s}$) | **143.8 FPS** | **24.8 FPS** | Baseline easily exceeds $15\text{ m/s}$ line rate |
| **Real-time SLA Met?** | Yes / No | **PASSED** ($6.95\text{ ms} \ll 50\text{ ms}$) | **PASSED** ($40.28\text{ ms} \le 50\text{ ms}$) | Both meet real-time threshold |

### The Industrial Engineering Dilemma
- `BaselineDefectCNN` operates at an extraordinary **144 FPS** on CPU with negligible power draw, but misclassified 15 low-contrast blemish samples (mutual confusion between slag inclusions and shallow pits).
- `Pretrained ResNet-18` eliminated nearly all confusion ($99.63\%$ accuracy, only 1 error across 270 test samples), but its **24.8 FPS** CPU throughput cannot maintain high line speeds ($>15\text{ m/s}$) without expensive GPU server clusters.

---

## 5. Proposed Architecture Innovation: `SE-BaselineDefectCNN`

To resolve this trade-off, we formulated a testable, student-achievable innovation hypothesis ([`docs/innovation_hypothesis.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/innovation_hypothesis.md)):

```
Input [H x W x C] ---> Squeeze: GAP ---> z [1 x 1 x C] ---> Bottleneck MLP (r=4) ---> s [1 x 1 x C]
      |                                                                                    |
      +-------------------------------- [ Channel Multiply: s_c * u_c ] <------------------+
                                                      |
                                                      v
                                        Recalibrated Output U~ [H x W x C]
```

- **Mechanism:** Insert Squeeze-and-Excitation (SE) channel-attention modules ($r=4$) after each Conv-BN-ReLU stage.
- **Hypothesis ($H_1$):** Squeeze-and-Excitation dynamically amplifies defect-salient micro-edge channels and suppresses background roller striations, driving Macro F1 from **$94.48\% \to \ge 96.5\%$** and boosting Inclusion F1 to **$\ge 93.5\%$**, while adding only **2,744 parameters (+9.7%)** and maintaining CPU latency **$< 7.5\text{ ms}$ ($>130\text{ FPS}$)**.
- **Controlled Protocol:** 4-question framework (What changed? Why? What happened? Did it improve?) with McNemar's significance testing ($p < 0.05$) locked into an auditable invariance matrix ([`docs/controlled_experiment_plan.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/controlled_experiment_plan.md)).

---

## 6. Two-Tier Factory Deployment Architecture

Rather than choosing between edge throughput and deep server accuracy, we propose an operational **Two-Tier Inspection Architecture**:

```
[ Moving Steel Strip (15 m/s) ]
               |
               v
  [ High-Speed Line-Scan Camera ]
               |
               +-------------------------------------------------------+
               |                                                       |
               v                                                       v
  [ TIER 1: Edge Gatekeeper ]                             [ Optical Frame Stream ]
  - Hardware: Embedded Camera DSP / Intel Atom                         |
  - Model: Baseline / SE-BaselineCNN (144 FPS)                         |
  - Latency: 6.95 ms                                                   |
  - Logic: Compute Defect Confidence Score (c)                         |
               |                                                       |
       +-------+-------+                                               |
       |               |                                               |
  c >= 0.95       c < 0.95 (Low Confidence / Flagged)                  |
       |               |                                               |
       v               v                                               |
    [ PASS ]     [ TIER 2: Central Defect Resolver ] <-----------------+
                 - Hardware: Edge Server (GPU / High-Core CPU)
                 - Model: Pretrained ResNet-18 (99.63% Accuracy)
                 - Latency: 40.28 ms
                 - Output: Defect Category & Pneumatic Sorter Actuation
                               |
                       +-------+-------+
                       |               |
                    [ REVIEW ]     [ REJECT ]
```

---

## 7. Scope Boundary & Phase II Technical Roadmap

```
+----------------------------------------------------------------------------------------------------+
|                                    PROJECT SCOPE BOUNDARY                                          |
|  PHASE I (COMPLETED): Image-level 6-class classification, neural foundations from scratch,          |
|  baseline CNN, controlled ablations, transfer learning benchmark, and innovation plan.             |
|                                                                                                    |
|  PHASE II (FUTURE): Object detection bounding boxes, pixel segmentation, generative augmentation,  |
|  and production web dashboard. (STRICTLY RESERVED FOR PHASE II - DO NOT IMPLEMENT IN PHASE I).     |
+----------------------------------------------------------------------------------------------------+
```

### Phase II Milestones
1. **Defect Localization (Bounding Boxes):** Implement YOLOv8 and RetinaNet to predict spatial coordinates `[x_min, y_min, x_max, y_max]` for multi-defect coils.
2. **Defect Quantification (Pixel Segmentation):** Deploy U-Net / SegNet to compute precise square-millimeter defect area and pit depth for scrap grading.
3. **Generative Data Expansion:** Develop Conditional Diffusion Models / GANs to synthesize rare defect samples (crazing fissures) to augment training distributions.
4. **Industrial Edge UI & Telemetry:** Build a responsive full-stack operator dashboard with real-time RTSP video streaming, alarm thresholds, and shift audit logs.
5. **Hardware TensorRT Optimization:** Quantize models to INT8 precision via NVIDIA TensorRT, targeting $< 1.0\text{ ms}$ inference on Jetson Orin edge boards.

---

## 8. Master Phase I Verification & Sign-Off Checklist

| Verification Item | Phase I Target / Requirement | Achieved Status | Auditable Artifact Reference |
| :--- | :--- | :--- | :--- |
| **Industrial Problem Definition** | Clear operational SLA and 6 classes | **COMPLETED** | [`docs/industrial_use_case.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/industrial_use_case.md) |
| **Dataset Selection & Audit** | NEU dataset (1,800 images, 6 classes) | **AUDITED & LOGGED** | [`docs/dataset_audit_report.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/dataset_audit_report.json) |
| **Leakage-Safe Splits** | 70/15/15 stratified splits (seed=42) | **ISOLATED & ZERO LEAK**| `data/splits/train.csv`, `val.csv`, `test.csv` |
| **Foundations from Scratch** | GEMM, SVD, Perceptrons, MLP backprop | **VALIDATED ($10^{-11}$)** | [`src/mlp_scratch.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/mlp_scratch.py), [`docs/tensor_sandbox.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/tensor_sandbox.md) |
| **Baseline Architecture** | Custom CNN $< 30\text{K}$ parameters | **BUILT (28,358 params)**| [`src/baseline_cnn.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/baseline_cnn.py) |
| **Baseline Accuracy SLA** | Macro F1 $\ge 85.0\%$, Acc $\ge 85.0\%$ | **EXCEEDED (94.44% Acc)** | [`docs/baseline_test_metrics.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/baseline_test_metrics.json) |
| **Baseline Latency SLA** | CPU Latency $\le 50.0\text{ ms}$ | **EXCEEDED (6.95 ms, 144 FPS)**| [`docs/baseline_test_metrics.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/baseline_test_metrics.json) |
| **Controlled Ablations** | 5 controlled training conditions | **EXECUTED & ANALYZED** | [`docs/ablation_experiments.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/ablation_experiments.json) |
| **Transfer Learning Benchmark** | Advanced vision model comparison | **99.63% ACC (ResNet-18)** | [`docs/model_comparison_metrics.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/model_comparison_metrics.json) |
| **Innovation Hypothesis** | Testable edge channel attention design | **FORMULATED (SE-Baseline)**| [`docs/innovation_hypothesis.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/innovation_hypothesis.md) |
| **Controlled Experiment Plan**| 4-Question Framework & Decision Gates | **STANDARDIZED** | [`docs/controlled_experiment_plan.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/controlled_experiment_plan.md) |
| **Master Experiment Registry** | Consolidated 17 Phase I runs | **AUDITABLE LOGGING** | [`docs/experiment_log.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/experiment_log.md) |
| **Literature & Contribution** | Formal citations & human vs AI matrix | **DOCUMENTED & ETHICAL** | [`docs/source_contribution_log.md`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/source_contribution_log.md) |
| **Version Control & Remote** | Clean conventional commits pushed | **PUSHED TO MAIN** | GitHub Commit: `f49970d` |

---
*End of Phase I Executive Project Proposal (Step 27)*
