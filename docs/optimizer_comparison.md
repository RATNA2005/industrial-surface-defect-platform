# Controlled Optimizer Comparison: Batch GD vs. SGD vs. Momentum vs. Nesterov

**Implementation Module:** [`src/optimizer_comparison.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/optimizer_comparison.py)  
**Machine-Readable Metrics:** [`docs/optimizer_comparison_results.json`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/optimizer_comparison_results.json)  
**Comparison Curves:** [`docs/figures/optimizer_comparison.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/optimizer_comparison.png)

---

## 1. Mathematical Formulations

### 1.1 Batch Gradient Descent (BGD)
Computes the exact gradient of the loss over the entire dataset of size $N$:
$$\mathbf{g}_t = \frac{1}{N} \nabla_{\theta} \sum_{i=1}^N \mathcal{L}(f(\mathbf{x}_i; \theta_t), y_i)$$
$$\theta_{t+1} = \theta_t - \eta \mathbf{g}_t$$
- **Property:** Deterministic, smooth, monotonic descent.
- **Drawback:** Only 1 parameter update per epoch; computationally slow on large industrial datasets.

### 1.2 Mini-Batch Stochastic Gradient Descent (SGD)
Partitions the dataset into mini-batches of size $B = 16$:
$$\mathbf{g}_t = \frac{1}{B} \nabla_{\theta} \sum_{i \in \mathcal{B}} \mathcal{L}(f(\mathbf{x}_i; \theta_t), y_i)$$
$$\theta_{t+1} = \theta_t - \eta \mathbf{g}_t$$
- **Property:** Performs $N / B$ parameter updates per epoch. High variance helps escape shallow local minima.
- **Drawback:** Oscillates erratically across steep ravines.

### 1.3 Classical Momentum (Polyak, 1964)
Introduces an exponential moving average of past gradients (velocity $\mathbf{v}_t$) with friction factor $\gamma = 0.85$:
$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \mathbf{g}_t$$
$$\theta_{t+1} = \theta_t - \mathbf{v}_{t+1}$$
- **Property:** Cancels out cross-ravine oscillations and accelerates along persistent gradient directions.

### 1.4 Nesterov Accelerated Gradient (NAG) (Nesterov, 1983)
Evaluates the gradient at a predictive "look-ahead" parameter position $\theta_t - \gamma \mathbf{v}_t$ rather than the current position:
$$\mathbf{g}_t^{\text{lookahead}} = \nabla_{\theta} \mathcal{L}(\theta_t - \gamma \mathbf{v}_t)$$
$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \mathbf{g}_t^{\text{lookahead}}$$
$$\theta_{t+1} = \theta_t - \mathbf{v}_{t+1}$$
- **Property:** Anticipatory braking mechanism that reduces overshooting when approaching a minimum.

---

## 2. Controlled Experimental Setup

To ensure strict scientific rigor, all four optimizers were evaluated under strictly identical conditions:
- **Dataset:** 200 synthetic surface inspection samples with non-linear curved defect boundaries.
- **Architecture:** Multilayer Perceptron ($2 \to 8 \to 1$) with Sigmoid activation.
- **Weight Initialization:** Exact identical parameters via deep copy of base model (`seed = 42`).
- **Learning Rate:** $\eta = 0.4$ across all optimizers.
- **Batch Size:** $B = 16$ (for SGD, Momentum, and NAG).
- **Epoch Count:** 120 epochs.

---

## 3. Empirical Results & Performance Comparison

| Optimizer | Updates / Epoch | Initial Loss | Final Loss (Epoch 120) | Final Accuracy | Epochs to Loss $< 0.1$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Batch GD** | 1 | 0.7685 | 0.23904 | 99.5% | Did not reach ($> 120$) |
| **Mini-batch SGD** | 13 | 0.6480 | 0.01975 | 99.5% | 19 epochs |
| **Momentum** | 13 | 0.5224 | **0.00629** | **100.0%** | **3 epochs** |
| **Nesterov (NAG)** | 13 | 0.5497 | **0.00629** | **100.0%** | **3 epochs** |

---

## 4. Key Observations & Optimization Dynamics

1. **Batch GD is Too Slow for Industrial Deep Learning:**  
   Because Batch GD only updates weights once per epoch, after 120 epochs it had executed only 120 updates and had a final loss of $0.239$. It failed to reach the loss $< 0.1$ threshold.
2. **Mini-Batch SGD Accelerates Progress by 13x:**  
   With $B = 16$, SGD executed $13 \times 120 = 1,560$ updates in the same 120 epochs, reaching loss $< 0.1$ in 19 epochs and achieving a final loss of $0.01975$.
3. **Momentum and Nesterov Provide Dramatic Acceleration:**  
   Both Momentum and Nesterov reached the loss $< 0.1$ threshold in just **3 epochs** (over $6\times$ faster than mini-batch SGD and $> 40\times$ faster than Batch GD). Their accumulated velocity propelled them straight down the non-linear curvature without stalling in shallow plateaus.
4. **Visual Evidence:**  
   The log-loss comparison plot ([`docs/figures/optimizer_comparison.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/optimizer_comparison.png)) shows the exponential drop of Momentum and NAG compared to the sluggish linear descent of Batch GD.
