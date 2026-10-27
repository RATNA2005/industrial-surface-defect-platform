# Baseline CNN Training Protocol & Experimental Record

**Implementation Module:** [`src/train_baseline.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/train_baseline.py)  
**Trained Model Checkpoint:** [`models/baseline_cnn_best.pth`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/models/baseline_cnn_best.pth)  
**Machine-Readable Training Log:** [`docs/baseline_training_log.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/baseline_training_log.json)

---

## 1. Justification of Major Hyperparameter & Design Choices

| Parameter / Choice | Configured Value | Engineering Justification ("WHY") |
| :--- | :---: | :--- |
| **Model Architecture** | `BaselineDefectCNN` | Lightweight 3-stage modular CNN with Global Average Pooling (28,358 parameters) to prevent overfitting on 1,260 training images. |
| **Loss Function** | `CrossEntropyLoss` | Categorical Cross-Entropy combines `log_softmax` and Negative Log-Likelihood (`NLLLoss`) in a numerically stable manner for mutually exclusive 6-class classification. |
| **Optimizer** | `Adam` ($\beta_1=0.9, \beta_2=0.999$) | Automatically maintains per-parameter adaptive learning rates via 1st and 2nd gradient moments, converging significantly faster on visual tasks than un-tuned SGD. |
| **Initial Learning Rate** | $\eta = 1.0 \times 10^{-3}$ | Standard default for Adam with Kaiming-initialized CNNs; balances fast initial error reduction with gradient stability. |
| **LR Scheduler** | `ReduceLROnPlateau` | Monitors validation loss; cuts learning rate by $50\%$ (`factor=0.5`) if validation loss plateaus for 3 epochs (`patience=3`), enabling fine-grained convergence. |
| **Weight Decay ($L_2$)** | $\lambda = 1.0 \times 10^{-4}$ | Constrains $L_2$ norm of weights to penalize overly complex filter representations without suffocating learning. |
| **Batch Size** | $B = 32$ | Balances GPU/CPU cache efficiency with gradient estimation variance, yielding 40 stochastic gradient updates per epoch. |
| **Max Epochs & Checkpointing** | 20 Epochs | Provides sufficient iterations ($20 \times 40 = 800$ updates) to reach convergence while automatically saving the weights with minimum validation loss. |
| **Random Seed** | `seed = 42` | Strictly controls pseudo-random number generators across PyTorch, NumPy, and Python for 100% exact reproduction. |
| **Execution Hardware** | CPU (`torch.device('cpu')`) | Ensures the training pipeline is reproducible on any standard student laptop or workstation without requiring dedicated GPUs. |

---

## 2. Epoch-by-Epoch Empirical Training Log

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Current Learning Rate | Epoch Runtime | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 1.61196 | 50.00% | 1.69825 | 38.89% | $1.0 \times 10^{-3}$ | 38.0s | Checkpoint Saved |
| **2** | 0.90099 | 65.48% | 0.66800 | 71.11% | $1.0 \times 10^{-3}$ | 33.8s | Checkpoint Saved |
| **3** | 0.73556 | 69.84% | 0.66957 | 69.63% | $1.0 \times 10^{-3}$ | 31.9s | — |
| **4** | 0.65953 | 74.05% | 0.49461 | 78.52% | $1.0 \times 10^{-3}$ | 28.0s | Checkpoint Saved |
| **5** | 0.58242 | 76.75% | 0.41946 | 84.07% | $1.0 \times 10^{-3}$ | 28.7s | Checkpoint Saved |
| **6** | 0.55940 | 78.49% | 0.43973 | 82.96% | $1.0 \times 10^{-3}$ | 29.2s | — |
| **7** | 0.52324 | 79.84% | 0.45096 | 81.48% | $1.0 \times 10^{-3}$ | 29.2s | — |
| **8** | 0.45789 | 81.83% | 0.55607 | 76.30% | $1.0 \times 10^{-3}$ | 28.4s | — |
| **9** | 0.45116 | 82.62% | 0.28268 | 89.26% | $1.0 \times 10^{-3}$ | 31.9s | Checkpoint Saved |
| **10** | 0.43549 | 84.37% | 0.33503 | 85.93% | $1.0 \times 10^{-3}$ | 30.9s | — |
| **11** | 0.40033 | 85.56% | 0.38221 | 81.85% | $1.0 \times 10^{-3}$ | 27.8s | — |
| **12** | 0.39288 | 85.95% | 0.38627 | 83.70% | $1.0 \times 10^{-3}$ | 29.8s | — |
| **13** | 0.37219 | 87.22% | 0.24938 | 90.74% | $1.0 \times 10^{-3}$ | 28.7s | Checkpoint Saved |
| **14** | 0.35848 | 87.06% | 0.30799 | 88.52% | $1.0 \times 10^{-3}$ | 27.3s | — |
| **15** | 0.33579 | 87.78% | 0.25957 | 89.63% | $1.0 \times 10^{-3}$ | 26.7s | — |
| **16** | 0.33701 | 88.57% | 0.27300 | 89.63% | $1.0 \times 10^{-3}$ | 26.9s | — |
| **17** | 0.34636 | 86.43% | 0.32313 | 86.67% | $1.0 \times 10^{-3}$ | 27.0s | (Plateau detected) |
| **18** | 0.29479 | 89.44% | **0.20132** | **91.48%** | $5.0 \times 10^{-4}$ | 36.6s | **BEST CHECKPOINT SAVED** |
| **19** | 0.27711 | 91.03% | 0.21614 | **92.96%** | $5.0 \times 10^{-4}$ | 39.0s | — |
| **20** | 0.24574 | 91.19% | 0.22194 | 92.22% | $5.0 \times 10^{-4}$ | 49.7s | — |

---

## 3. Training Outcome Summary

- **Total Execution Wall Time:** $629.7\text{ seconds}$ ($\approx 10.5\text{ minutes}$ on standard CPU).
- **Best Validation Performance (Epoch 18):**
  - **Validation Loss:** **$0.20132$**
  - **Validation Accuracy:** **$91.48\%$** (Peak accuracy reached $92.96\%$ in Epoch 19).
- **Generalization Behavior:** Training loss ($0.24574$) and Validation loss ($0.20132$) track closely with zero severe overfitting divergence.
- **Criterion Check:** Exceeds the Phase I milestone target ($\text{Accuracy} \ge 85\%$).
