# Multilayer Perceptron (MLP) from Scratch: Architecture, Equations & Backpropagation

**Implementation Module:** [`src/mlp_scratch.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/mlp_scratch.py)  
**Framework:** Pure Python + NumPy (0% automatic differentiation used)

---

## 1. Network Architecture

The network is a 2-layer feedforward neural network designed for non-linear pattern classification:

```
Input Layer (X)           Hidden Layer (h)           Output Layer (y_hat)
[Dim: N x D_in]          [Dim: N x D_h]              [Dim: N x D_out]
       ○ ────────────┐
       ○ ───────────┐│ ───►  ○ ────────┐
                     ││      ○ ────────┼───►  ○ ──► y_hat ∈ (0, 1)
                     ││      ○ ────────┘
       ○ ────────────┘│
     (D_in = 2)           (D_h = 4 or 8)            (D_out = 1)
```

---

## 2. Parameters vs. Hyperparameters

A core distinction in deep learning theory:

| Category | Definition | Role in Our MLP | Examples in This Implementation |
| :--- | :--- | :--- | :--- |
| **Parameters** | Internal variables learned and updated automatically by the model during training via backpropagation. | Define the exact transformation matrices and bias shifts. | • Weight matrices: $W_1 \in \mathbb{R}^{2 \times 4}$, $W_2 \in \mathbb{R}^{4 \times 1}$<br>• Bias vectors: $b_1 \in \mathbb{R}^{1 \times 4}$, $b_2 \in \mathbb{R}^{1 \times 1}$<br>*(Total 17 learnable scalar parameters)* |
| **Hyperparameters** | External configuration knobs set by the machine learning engineer before training begins. | Control the model capacity, optimization dynamics, and training speed. | • Learning rate ($\eta = 2.0$)<br>• Epoch count ($T = 2000$)<br>• Hidden units ($D_h = 4$ or $8$)<br>• Activation function ($\text{Sigmoid}$)<br>• Weight init scheme ($\text{Xavier/Glorot}$)<br>• Random seed (`seed = 42`) |

---

## 3. Mathematical Forward Propagation

For an input batch $X \in \mathbb{R}^{N \times D_{\text{in}}}$:

1. **Hidden Pre-Activation:**
   $$Z_1 = X W_1 + \mathbf{b}_1 \quad \in \mathbb{R}^{N \times D_h}$$
2. **Hidden Activation (Sigmoid):**
   $$A_1 = \sigma(Z_1) = \frac{1}{1 + e^{-Z_1}} \quad \in \mathbb{R}^{N \times D_h}$$
3. **Output Pre-Activation:**
   $$Z_2 = A_1 W_2 + \mathbf{b}_2 \quad \in \mathbb{R}^{N \times D_{\text{out}}}$$
4. **Output Activation / Prediction:**
   $$\hat{Y} = A_2 = \sigma(Z_2) \quad \in \mathbb{R}^{N \times D_{\text{out}}}$$
5. **Binary Cross-Entropy Loss:**
   $$\mathcal{L}(W, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

---

## 4. Analytical Backpropagation (Multivariate Chain Rule)

We compute the exact partial derivatives of the scalar loss $\mathcal{L}$ with respect to all four parameters ($W_1, \mathbf{b}_1, W_2, \mathbf{b}_2$):

### Step 1: Output Error Term ($\delta_2$)
$$\delta_2 = \frac{\partial \mathcal{L}}{\partial Z_2} = \frac{\partial \mathcal{L}}{\partial A_2} \odot \sigma'(Z_2) = \left[ -\frac{y}{A_2} + \frac{1 - y}{1 - A_2} \right] \odot [A_2 \odot (1 - A_2)] = \frac{1}{N} (A_2 - Y)$$

### Step 2: Output Layer Parameter Gradients
$$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \delta_2 \quad \in \mathbb{R}^{D_h \times D_{\text{out}}}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = \sum_{i=1}^N (\delta_2)_i \quad \in \mathbb{R}^{1 \times D_{\text{out}}}$$

### Step 3: Hidden Error Term ($\delta_1$)
Backpropagating the error through $W_2$ into the hidden layer:
$$\delta_1 = \frac{\partial \mathcal{L}}{\partial Z_1} = (\delta_2 W_2^T) \odot \sigma'(Z_1) = (\delta_2 W_2^T) \odot [A_1 \odot (1 - A_1)] \quad \in \mathbb{R}^{N \times D_h}$$

### Step 4: Hidden Layer Parameter Gradients
$$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \delta_1 \quad \in \mathbb{R}^{D_{\text{in}} \times D_h}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \sum_{i=1}^N (\delta_1)_i \quad \in \mathbb{R}^{1 \times D_h}$$

### Step 5: Parameter Updates (Gradient Descent)
$$W_1 \leftarrow W_1 - \eta \frac{\partial \mathcal{L}}{\partial W_1}, \quad \mathbf{b}_1 \leftarrow \mathbf{b}_1 - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1}$$
$$W_2 \leftarrow W_2 - \eta \frac{\partial \mathcal{L}}{\partial W_2}, \quad \mathbf{b}_2 \leftarrow \mathbf{b}_2 - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}_2}$$

---

## 5. Empirical Verification Results

### 5.1 Solving the Non-Linear XOR Problem
A single-layer perceptron mathematically cannot solve XOR because the classes are not linearly separable. Our 2-layer MLP solves it completely:

| $x_1$ | $x_2$ | Target $y$ | MLP Output $\hat{y}$ | Prediction | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | **0.07%** | 0 | **[CORRECT]** |
| 0 | 1 | 1 | **99.78%** | 1 | **[CORRECT]** |
| 1 | 0 | 1 | **99.80%** | 1 | **[CORRECT]** |
| 1 | 1 | 0 | **0.33%** | 0 | **[CORRECT]** |

*Loss converged from $0.725192 \to 0.002067$ with **$100.0\%$ Accuracy**.*

### 5.2 Synthetic Non-Linear Defect Boundary
Trained an MLP ($2 \to 8 \to 1$) on concentric circular data (sound metal core surrounded by a defect scale ring):
- Final classification accuracy: **$100.0\%$**.
- Generated visual artifact: [`docs/figures/mlp_decision_boundary.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/mlp_decision_boundary.png).
