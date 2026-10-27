# Activation Functions: Mathematical Comparison & Architectural Justification

**Implementation Module:** [`src/activation_functions.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/activation_functions.py)  
**Comparison Figure:** [`docs/figures/activation_functions_comparison.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/activation_functions_comparison.png)

---

## 1. Master Comparative Matrix

| Function | Equation $f(z)$ | First-Order Derivative $f'(z)$ | Output Range | Differentiability | Saturation Properties | Primary Advantages | Primary Limitations | Role in Platform |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1. Step** | $\begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$ | $0$ ($\forall z \ne 0$) | $\{0, 1\}$ (Discrete) | **Non-differentiable** | Saturated everywhere ($f'=0$) | Historical biological realism; rigid binary thresholding. | **Zero gradient** halts backpropagation. Cannot learn. | Educational baseline (M-P neuron). |
| **2. Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | $(0, 1)$ (Continuous) | Differentiable everywhere | Saturated for $\|z\| > 4$; max gradient $0.25$ at $z=0$. | Smooth continuous derivative; probabilistic interpretation. | **Vanishing gradient** in deep layers; non-zero centered output. | Binary defect confidence / Multi-label gates. |
| **3. Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | $(-1, 1)$ (Continuous) | Differentiable everywhere | Saturated for $\|z\| > 3$; max gradient $1.0$ at $z=0$. | **Zero-centered** output avoids zig-zag gradient updates; steeper gradient than sigmoid. | Still suffers from vanishing gradients when saturated. | Recurrent / feature embedding layers. |
| **4. ReLU** | $\max(0, z)$ | $\begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$ | $[0, \infty)$ (Unbounded) | Differentiable for $z \ne 0$ (subgradient at 0) | **Non-saturating** on positive half-plane ($f'=1$). | **Solves vanishing gradient**; computationally fast ($O(1)$ max); sparse activations. | **Dying ReLU problem** (neurons inactive for $z < 0$); non-zero centered. | **Default choice for CNN hidden layers.** |

---

## 2. In-Depth Mathematical Analysis

### 2.1 The Vanishing Gradient Problem (Sigmoid & Tanh)
When backpropagating an error through $L$ hidden layers with sigmoid activations, the gradient receives repeated multiplications of the activation derivative:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_1} \propto \prod_{l=1}^L \sigma'(z_l)$$
Because $\max \sigma'(z) = 0.25$, after just 4 layers the gradient magnitude is scaled by at most:
$$0.25^4 = 0.0039$$
After 8 layers, the gradient is $< 0.000015$. The early layers of a deep network receive virtually zero updates and freeze.

### 2.2 Why ReLU Solves the Vanishing Gradient
For all positive pre-activations ($z > 0$), the derivative of ReLU is strictly $1.0$:
$$\frac{d}{dz} \text{ReLU}(z) = 1.0 \quad (\forall z > 0)$$
Consequently, error gradients propagate backward through arbitrarily deep convolutional stacks without decaying exponentially.

### 2.3 The "Dying ReLU" Phenomenon & Mitigation
If a large negative gradient update knocks a neuron's bias so low that $z = \mathbf{w}^T \mathbf{x} + b < 0$ for all training samples, its output becomes permanently $0.0$ and its gradient becomes permanently $0.0$. The neuron effectively "dies" and cannot recover.
* **Mitigations in Modern CNNs:** Small learning rates, He/Kaiming initialization (which accounts for the fact that half of activations are zeroed), and batch normalization.

---

## 3. Architectural Justification for Step 16 Baseline CNN

Based on the empirical properties evaluated in this step:
1. **Convolutional & Dense Hidden Layers $\to$ Use ReLU:**  
   Ensures rapid, stable gradient flow across the convolutional feature extractor, avoiding vanishing gradients on $200 \times 200$ images.
2. **Classification Output Layer $\to$ Use Softmax:**  
   Normalizes the 6 unconstrained final logits $\mathbf{z} \in \mathbb{R}^6$ into a mutually exclusive posterior probability distribution $\sum_{i=1}^6 P(\text{Class}_i) = 1.0$ for industrial defect classification.
