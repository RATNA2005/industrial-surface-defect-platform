# Tensor Mathematics Educational Sandbox

**Implementation Module:** [`src/tensor_sandbox.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/tensor_sandbox.py)  
**Verification Status:** 100% Passed (All manual implementations verified against NumPy with error $< 10^{-11}$)

---

## 1. Tensor Orders & Dimensional Hierarchy in Visual Inspection

In the **Industrial Surface Defect Inspection Platform**, data flows through five tensor orders:

| Tensor Order | Rank | Mathematical Object | Inspection Platform Analogy | Shape in Our Pipeline |
| :---: | :---: | :--- | :--- | :---: |
| **0D** | 0 | **Scalar** | Batch Cross-Entropy Loss value $\mathcal{L}$ | `()` |
| **1D** | 1 | **Vector** | Class bias parameters $\mathbf{b}$ or 6-class probability vector | `(6,)` |
| **2D** | 2 | **Matrix** | Fully-connected dense layer weight matrix $W$ | `(4, 6)` |
| **3D** | 3 | **Order-3 Tensor** | Single preprocessed surface image crop $(C, H, W)$ | `(3, 200, 200)` |
| **4D** | 4 | **Order-4 Tensor** | Mini-batch of surface images $(B, C, H, W)$ | `(32, 3, 200, 200)` |

---

## 2. Mathematical Operations & Manual Verifications

### 2.1 Broadcasting
* **Rule:** Two tensors are compatible for broadcasting if trailing axes match or have length 1.
* **Equation:**
  $$C_{i, j} = A_{i, j} + b_j \quad \text{for } i \in [0, N-1], j \in [0, M-1]$$
* **Verification:** Manual nested loop vs `A + b` yielded **$\text{Max Absolute Error} = 0.0$**.

### 2.2 Dot Product & Cosine Similarity
* **Inner Product:**
  $$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^D u_i v_i$$
* **Cosine Similarity:**
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
* **Verification:** Manual summation vs `np.dot(u, v)` yielded **$\text{Absolute Error} = 0.0$**.

### 2.3 General Matrix Multiplication (GEMM)
* **Equation:**
  $$Y_{i, k} = \left(\sum_{j=1}^M X_{i, j} W_{j, k}\right) + b_k$$
* **Verification:** Manual triple-nested loop ($O(N \cdot M \cdot P)$) vs `np.matmul(X, W) + b` yielded **$\text{Max Absolute Error} = 1.1 \times 10^{-16}$** (floating-point precision limit).

### 2.4 Numerically Stable Softmax
* **Equation with Stability Shift:**
  $$\text{Softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^K e^{z_j - \max(\mathbf{z})}}$$
* **Why Needed:** Without subtracting $\max(\mathbf{z})$, logits $> 709$ cause IEEE 754 floating-point overflow (`inf`), producing `NaN` loss during backpropagation.

---

## 3. Singular Value Decomposition (SVD) on Defect Surfaces

* **Decomposition:**
  $$A = U \Sigma V^T = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$
  where $A \in \mathbb{R}^{200 \times 200}$ is an industrial steel surface crop (`scratches_1.jpg`).
* **Low-Rank Approximation ($k$-Rank Truncation):**
  $$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$
* **Energy Retained:**
  $$\text{Energy Ratio} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{200} \sigma_i^2} \times 100\%$$
* **Empirical Energy Retained:**
  - **Rank $k=1$:** $98.8\%$ energy retained (captures global surface illumination).
  - **Rank $k=5$:** $99.4\%$ energy retained (captures prominent macro-scratch orientation).
  - **Rank $k=20$:** $99.7\%$ energy retained (nearly indistinguishable from the raw crop; high-frequency texture preserved).
  - **Rank $k=50$:** $99.9\%$ energy retained.
* **Figure:** [`docs/figures/svd_reconstruction.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/svd_reconstruction.png)

---

## 4. Numerical Gradient Checking (Finite Differences)

* **Mathematical Formula (Two-Sided Central Difference):**
  $$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta + \epsilon \mathbf{e}_i) - \mathcal{L}(\theta - \epsilon \mathbf{e}_i)}{2\epsilon}$$
  where $\epsilon = 10^{-5}$.
* **Relative Error Metric:**
  $$\text{Relative Error} = \frac{\|\nabla_{\text{num}} - \nabla_{\text{analytical}}\|_2}{\|\nabla_{\text{num}}\|_2 + \|\nabla_{\text{analytical}}\|_2}$$
* **Empirical Verification Result:**
  $$\text{Relative Error} = 1.06 \times 10^{-11} \ll 10^{-7}$$
* **Verdict:** Backpropagation analytical gradients are mathematically verified with machine precision.
