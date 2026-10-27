# Controlled Experiment Plan: Evaluating the SE-BaselineDefectCNN Innovation

**Document Version:** 1.0.0  
**Phase:** Phase I — Deep Learning Foundations & Architecture Innovation (Step 24)  
**Author:** Ratna Khanna (`ratnakhanna2005@gmail.com`)  
**Project:** Industrial Surface Defect Inspection Platform (Hot-Rolled Steel Strip Surface Defect Inspection)  
**Target Innovation:** `SE-BaselineDefectCNN` (Squeeze-and-Excitation Channel Attention, $r=4$)  
**Baseline Control:** `BaselineDefectCNN` (From-Scratch 3-Stage CNN, Step 18 Checkpoint)  
**Evaluation Dataset:** NEU Surface Defect Database Held-Out Test Partition (`data/splits/test.csv`, $N=270$)  

---

## 1. Executive Summary & Protocol Overview

In Step 23, we formulated the formal Innovation Hypothesis ($H_1$) proposing that incorporating **Squeeze-and-Excitation (SE) channel-attention recalibration blocks** into our lightweight baseline architecture will selectively amplify defect-salient micro-texture features while suppressing background rolling striations. 

In Step 24, we formalize the **Controlled Experiment Plan**. Following rigorous empirical scientific methodology, this document defines the four foundational research questions:
1. **What Changed?** (The exact architectural modification).
2. **Why?** (The empirical and domain-specific engineering justification).
3. **What Happened?** (The empirical baseline control established under identical conditions).
4. **Did It Improve?** (The quantitative acceptance criteria and statistical decision rules to accept or reject $H_1$).

Furthermore, this plan establishes the **Invariance Matrix** guaranteeing that every extraneous variable (dataset partitions, data normalization, augmentation parameters, optimizer hyperparameters, learning rate schedules, loss function, and CPU execution hardware) is held strictly invariant across control and experimental runs.

---

## 2. The 4-Question Scientific Framework

```
+----------------------------------------------------------------------------------------------------+
|                                 CONTROLLED EXPERIMENT FRAMEWORK                                    |
+------------------------+------------------------+------------------------+-------------------------+
|   1. WHAT CHANGED?     |        2. WHY?         |   3. WHAT HAPPENED?    |   4. DID IT IMPROVE?    |
|   Architectural Diff   |  Domain Rationale      |  Baseline Control      |  Decision Rule & SLA    |
|                        |                        |                        |                         |
| - Insert 3 SE units    | - Baseline had 15 test | - 94.44% Test Accuracy | - Target: >= 96.5% F1   |
| - Reduction ratio r=4  |   errors (10 on hard)  | - 0.9448 Macro F1      | - Inclusions >= 93.5%   |
| - GAP Squeeze          | - Inclusions F1: 89.6% | - Latency: 6.95 ms     | - Pitted Surface >= 95% |
| - 2-layer MLP Sigmoid  | - Pitted Surf F1: 92.0%| - Params: 28,358       | - Params <= 31,500      |
| - +2,744 params (+9.7%)| - Low-contrast blemish | - Checkpoint: 356 KB   | - Latency <= 8.0 ms     |
+------------------------+------------------------+------------------------+-------------------------+
```

### 2.1 Question 1: What Changed? (The Independent Variable)
The sole manipulated variable between the control and experimental treatment is the addition of **Squeeze-and-Excitation (SE) channel attention modules** immediately following each of the three Conv-BN-ReLU stages in `BaselineDefectCNN`:

1. **Structural Insertion:**
   - **Stage 1:** A 16-channel SE block ($16 \to 4 \to 16$, reduction ratio $r=4$) is inserted between `ReLU1` and `MaxPool1`.
   - **Stage 2:** A 32-channel SE block ($32 \to 8 \to 32$, reduction ratio $r=4$) is inserted between `ReLU2` and `MaxPool2`.
   - **Stage 3:** A 64-channel SE block ($64 \to 16 \to 64$, reduction ratio $r=4$) is inserted between `ReLU3` and `MaxPool3`.
2. **Computational Mechanism:**
   - **Squeeze:** Global Average Pooling transforms spatial feature maps $\mathbf{u}_c \in \mathbb{R}^{H \times W}$ into channel descriptors $z_c = \frac{1}{HW} \sum_{i,j} u_c(i,j)$.
   - **Excitation:** A two-layer MLP bottleneck with ReLU non-linearity and Sigmoid output computes dynamic channel weights $\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{z})\right) \in [0, 1]^C$.
   - **Recalibration:** Channel-wise scalar multiplication scales each feature channel: $\tilde{\mathbf{x}}_c = s_c \cdot \mathbf{u}_c$.
3. **Parameter Footprint:**
   - Stage 1 SE: $(16 \times 4 + 4) + (4 \times 16 + 16) = \mathbf{152}$ parameters.
   - Stage 2 SE: $(32 \times 8 + 8) + (8 \times 32 + 32) = \mathbf{552}$ parameters.
   - Stage 3 SE: $(64 \times 16 + 16) + (16 \times 64 + 64) = \mathbf{2,192}$ parameters.
   - Total Attention Overhead: **2,896 parameters** ($+10.2\%$ over the 28,358 baseline parameters $\to \mathbf{31,254}$ total parameters).

### 2.2 Question 2: Why? (Scientific & Engineering Justification)
1. **Empirical Defect Ambiguity (Observed in Steps 18 & 22):**
   In our baseline evaluation on the 270 test samples ([`docs/baseline_test_metrics.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/baseline_test_metrics.json)), 10 out of the 15 total misclassifications occurred on two specific classes:
   - **Slag Inclusions (8 False Positives, 2 False Negatives, F1: 89.58%):** Confused with pitted surfaces and scratches.
   - **Pitted Surfaces (2 False Positives, 5 False Negatives, F1: 91.95%):** Confused with slag inclusions.
2. **Physical Rolling Mill Context:**
   Both defects manifest as dark, localized blemishes against the moving steel strip. Under standard spatial convolution, all feature channels are combined uniformly without regard to signal-to-noise ratio. Harmless background roller marks (longitudinal striations caused by work-roll wear) generate strong responses across certain channels, swamping the subtle, shallow gradient depressions of pitted surfaces.
3. **The Channel Attention Remedy:**
   Channel attention provides an adaptive filtering mechanism. Squeeze-and-Excitation allows the network to learn global energy signatures: when a frame contains shallow depressions, the excitation network assigns high weights ($s_c \to 1.0$) to gradient-shading channels and suppresses channels dominated by uniform roller striations ($s_c \to 0.0$).

### 2.3 Question 3: What Happened? (The Controlled Baseline Reference)
Under the identical training and evaluation protocol, the baseline model (`BaselineDefectCNN`, Step 18) produced the following auditable metrics on the held-out test partition ($N=270$):
- **Test Accuracy:** **94.44%** (255 / 270 correct).
- **Macro F1-Score:** **0.9448** (94.48%).
- **Macro Precision:** **0.9484** (94.84%).
- **Macro Recall:** **0.9444** (94.44%).
- **Inclusion F1-Score:** **0.8958** (89.58%).
- **Pitted Surface F1-Score:** **0.9195** (91.95%).
- **CPU Inference Latency ($B=1$):** **6.95 ms** (Mean), **8.44 ms** (P95).
- **Throughput:** **143.8 FPS** on CPU.
- **Model Checkpoint Size:** **355.7 KB** (28,358 parameters).

*This benchmark serves as the strict control reference condition ($C_0$).*

### 2.4 Question 4: Did It Improve? (Decision Rules & Statistical Testing)
To prevent confirmation bias, we define quantitative gates required to reject the null hypothesis ($H_0$) and declare the innovation successful ($H_1$):

| Acceptance Gate | Baseline Reference ($C_0$) | Target SLA Threshold | Minimum $\Delta$ Required | Verification Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1: Overall Macro F1** | $0.9448$ | **$\ge 0.9650$** (96.50%) | $\mathbf{\ge +2.02\%}$ | Reject $H_0$ if Macro F1 $\ge 0.9650$ |
| **Gate 2: Overall Accuracy** | $94.44\%$ | **$\ge 96.50\%$** (261/270) | $\mathbf{\ge +2.06\%}$ (Reduce errors $\le 9$) | Test Accuracy $\ge 96.50\%$ |
| **Gate 3: Inclusion F1-Score**| $0.8958$ | **$\ge 0.9350$** (93.50%) | $\mathbf{\ge +3.92\%}$ | Per-class test F1 $\ge 0.9350$ |
| **Gate 4: Pitted Surface F1** | $0.9195$ | **$\ge 0.9550$** (95.50%) | $\mathbf{\ge +3.55\%}$ | Per-class test F1 $\ge 0.9550$ |
| **Gate 5: Parameter Budget** | $28,358$ | **$\le 31,500$** | $\mathbf{\le +11.1\%}$ growth | Total parameters $\le 31,500$ |
| **Gate 6: Real-time Latency** | $6.95\text{ ms}$ | **$\le 8.00\text{ ms}$** | Real-time SLA ($\le 50\text{ ms}$) | Sustained throughput $\ge 125\text{ FPS}$ |
| **Gate 7: Statistical Significance** | N/A | **McNemar's Test $p < 0.05$** | Statistically significant | Paired contingency test on test set |

---

## 3. Controlled Variables Matrix

To ensure that any observed performance difference is caused solely by the introduction of the SE attention mechanism (and not by confounding factors), all parameters are locked according to the matrix below:

| Dimension | Controlled Variable | Baseline Control (`BaselineDefectCNN`) | Treatment (`SE-BaselineDefectCNN`) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Data Partition** | Training Set | `data/splits/train.csv` ($N=1,260$) | `data/splits/train.csv` ($N=1,260$) | **IDENTICAL** |
| | Validation Set | `data/splits/val.csv` ($N=270$) | `data/splits/val.csv` ($N=270$) | **IDENTICAL** |
| | Test Set | `data/splits/test.csv` ($N=270$) | `data/splits/test.csv` ($N=270$) | **IDENTICAL** |
| **Preprocessing** | Image Resolution | $200 \times 200$ pixels | $200 \times 200$ pixels | **IDENTICAL** |
| | Color Channels | 3-channel RGB (Grayscale encoded) | 3-channel RGB (Grayscale encoded) | **IDENTICAL** |
| | Normalization | $\mu_{\text{train}}=[0.504, 0.504, 0.504]$, $\sigma_{\text{train}}=[0.212, 0.212, 0.212]$ | Same training-derived $\mu, \sigma$ | **IDENTICAL** |
| **Augmentation** | RandomHorizontalFlip | $p=0.5$ | $p=0.5$ | **IDENTICAL** |
| | RandomVerticalFlip | $p=0.5$ | $p=0.5$ | **IDENTICAL** |
| | RandomAffine | degrees=10, translate=(0.05, 0.05) | degrees=10, translate=(0.05, 0.05) | **IDENTICAL** |
| **Optimization** | Optimizer | Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | **IDENTICAL** |
| | Initial Learning Rate | $\eta = 10^{-3}$ ($0.001$) | $\eta = 10^{-3}$ ($0.001$) | **IDENTICAL** |
| | Weight Decay | $10^{-4}$ ($0.0001$) | $10^{-4}$ ($0.0001$) | **IDENTICAL** |
| | Loss Function | CrossEntropyLoss (unweighted) | CrossEntropyLoss (unweighted) | **IDENTICAL** |
| | Batch Size | $B=32$ | $B=32$ | **IDENTICAL** |
| | Max Epochs | 20 epochs | 20 epochs | **IDENTICAL** |
| | LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) | ReduceLROnPlateau (factor=0.5, patience=3) | **IDENTICAL** |
| **Hardware** | Execution Device | Intel CPU (Single Thread, Batch=1) | Intel CPU (Single Thread, Batch=1) | **IDENTICAL** |
| | Timing Protocol | 100 warm-up runs + 270 timed passes | 100 warm-up runs + 270 timed passes | **IDENTICAL** |
| **Architecture** | **SE Channel Attention** | **NONE (Disabled)** | **ENABLED ($r=4$ across 3 stages)** | **TREATMENT** |

---

## 4. Visual Experiment Flowchart

The structured 4-question flowchart and execution sequence is visualized below:

![Controlled Experiment Flowchart](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/experiment_plan_flowchart.png)

*Figure 1: Visual mapping of the 4-question experimental protocol, connecting hypothesis motivation to rigorous 3-phase execution and decision gating.*

---

## 5. Step-by-Step Execution Protocol

```
+----------------------------------------------------------------------------------------------------+
|                                    EXPERIMENT EXECUTION SEQUENCE                                   |
+--------------------------+--------------------------+----------------------------------------------+
| Phase A: Implementation  | Phase B: Training Run    | Phase C: Evaluation & Hypothesis Gate        |
|                          |                          |                                              |
| 1. Build PyTorch modules | 1. 20 epochs on CPU      | 1. Evaluate on test.csv (N=270)              |
|    - SEBlock (r=4)       | 2. Adam (lr=1e-3, wd=1e-4)2. Benchmark latency (B=1, 270 timed passes)   |
|    - SEBaselineDefectCNN | 3. Validation monitoring | 3. Generate confusion matrix & class metrics  |
| 2. Parameter validation  | 4. Save best checkpoint  | 4. Conduct McNemar's significance test       |
|    (Target: ~31.2K)      |    to models/            | 5. Verify Gate 1 to Gate 7 decision criteria |
+--------------------------+--------------------------+----------------------------------------------+
```

### Phase A: Architecture Construction Specification
- Define `SEBlock(channels, reduction=4)`:
  - `fc1 = nn.Linear(channels, channels // reduction, bias=True)`
  - `relu = nn.ReLU(inplace=True)`
  - `fc2 = nn.Linear(channels // reduction, channels, bias=True)`
  - `sigmoid = nn.Sigmoid()`
  - Forward: $s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(x)))$, return $x \cdot s$.
- Define `SEBaselineDefectCNN`:
  - Exactly preserve Conv1 (16 ch), Conv2 (32 ch), Conv3 (64 ch) and Dense Head (64 $\to$ 64 $\to$ 6) from `BaselineDefectCNN`.
  - Insert `SEBlock` after each BN+ReLU block prior to MaxPool.

### Phase B: Training Protocol
- Seed random generators (`torch.manual_seed(42)`, `np.random.seed(42)`).
- Execute 20 epochs with `ReduceLROnPlateau`.
- Checkpoint saving criterion: Minimum Validation Loss ($\mathcal{L}_{\text{val}}$).
- Log training trajectory to `docs/se_baseline_training_log.json`.

### Phase C: Evaluation & Hypothesis Decision Protocol
- Load best checkpoint and set `.eval()`.
- Evaluate on `data/splits/test.csv` ($N=270$).
- Compute per-class precision, recall, F1, macro metrics, and confusion matrix.
- Profile single-sample CPU latency ($B=1$) with 100 discarded warmup runs.
- Log metrics to `docs/se_baseline_metrics.json`.
- Execute McNemar's Chi-squared test comparing baseline vs. SE predictions:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
  where $b$ is the number of samples correct in Baseline but incorrect in SE, and $c$ is the number of samples incorrect in Baseline but correct in SE.

---

## 6. Contingency Analysis & Scientific Risk Mitigation

| Contingency Scenario | Empirical Symptom | Scientific Root Cause | Remediation Protocol |
| :--- | :--- | :--- | :--- |
| **Scenario 1: Marginal Gain ($<+1.0\%$ F1)** | Macro F1 improves only from $94.48\% \to 95.2\%$ | Reduction ratio $r=4$ may over-compress or under-compress certain channels. | Execute controlled reduction ratio sweep: compare $r=2$ (richer bottleneck, $+5.6\text{K}$ params) vs. $r=8$ (tighter bottleneck, $+1.4\text{K}$ params). |
| **Scenario 2: Overfitting on Training Set** | Train accuracy reaches $99.8\%$ while Val accuracy stalls at $92.0\%$ | SE bottleneck weights memorizing sample-specific noise on small training corpus ($N=1,260$). | Increase classifier head dropout from $0.2 \to 0.3$ and increase weight decay from $10^{-4} \to 5 \times 10^{-4}$. |
| **Scenario 3: Latency Overhead $> 8.0\text{ ms}$** | CPU latency rises to $>10\text{ ms}$ due to un-vectorized operations. | PyTorch tensor reshaping overhead in GAP or excitation broadcast. | Optimize tensor operations using in-place broadcasting: `x * s.view(b, c, 1, 1)`. |

---

## 7. Viva Voce Examination Reference (Experiment Planning Topic)

### Q1: "Why did you only change ONE variable (adding SE blocks) instead of also tuning learning rate, batch size, or adding more layers?"
> **Answer:** "Adhering strictly to the **Single-Variable Scientific Method** is essential for causal attribution. If we simultaneously added SE blocks, changed the learning rate from $10^{-3}$ to $5\times 10^{-4}$, and increased batch size to 64, any observed performance increase could not be causally attributed to the channel attention mechanism. By locking every training, data, and optimizer parameter to match the Baseline control exactly, any statistically significant delta $\Delta$ in Macro F1 is provably caused by the attention mechanism."

### Q2: "Why use McNemar's Test instead of a standard two-sample Student's t-test for comparing model accuracies?"
> **Answer:** "A Student's t-test assumes independent, continuous, normally distributed samples. In model benchmarking, both models are evaluated on the **exact same held-out test partition** ($N=270$), violating the independence assumption. **McNemar's Test** is specifically designed for paired nominal data. It focuses on the discordant pairs—samples where Model A was correct and Model B was wrong ($b$), and vice-versa ($c$). The test statistic $\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$ evaluates whether the discrepancy between models is statistically significant ($p < 0.05$) rather than random noise."

### Q3: "How do you guarantee that no data leakage occurred during the experiment?"
> **Answer:** "We enforce strict three-way data isolation established in Step 6:
> 1. Stratified 70/15/15 splits were locked into deterministic CSV files (`train.csv`, `val.csv`, `test.csv`) with `seed=42`.
> 2. The exact duplicate image pair (`patches_101.jpg` and `patches_105.jpg` identified in Step 4) was confined entirely to the training split, preventing memorized test leakage.
> 3. Pixel normalization parameters ($\mu=0.5040, \sigma=0.2121$) were computed exclusively from the 1,260 training images and applied frozen to validation and test loaders.
> 4. The test partition was never evaluated during model selection or hyperparameter tuning."

---
*End of Controlled Experiment Plan Document (Step 24)*
