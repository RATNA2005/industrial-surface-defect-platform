# Sigmoid Neuron: Mathematical Foundations & Gradient Optimization

**Implementation Module:** [`src/sigmoid_neuron.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/sigmoid_neuron.py)  
**Verification Status:** 100% Passed (Gradient check relative error $4.30 \times 10^{-11} \ll 10^{-7}$)

---

## 1. Mathematical Formulation

```
Input Features (x_1, x_2, ..., x_n)
       ↓
Weights (w_1, w_2, ..., w_n) + Bias (b)
       ↓
Affine Transformation: z = Σ (w_i * x_i) + b = w^T x + b
       ↓
Sigmoid Activation: σ(z) = 1 / (1 + e^(-z))
       ↓
Continuous Probability Output: y_hat ∈ (0, 1)
```

### Definitions:
1. **Weighted Sum ($\sum w_i x_i$):** Linear combination of continuous real-valued inputs $\mathbf{x} \in \mathbb{R}^n$. It weights the relative importance of each feature (e.g. surface roughness vs. scratch contrast).
2. **Bias ($b$):** The learnable horizontal translation parameter. While the McCulloch-Pitts neuron utilized a rigid threshold $\theta$, the sigmoid neuron re-parameterizes this as $b = -\theta$, allowing the activation boundary to shift independently of the input origin.
3. **Sigmoid Function ($\sigma(z)$):** The logistic activation function mapping $(-\infty, \infty) \to (0, 1)$:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
4. **Output Range:** Continuous open interval $(0, 1)$. This provides a natural probabilistic interpretation: $\hat{y} = P(y = 1 \mid \mathbf{x})$, representing the model's confidence in surface defect classification.

---

## 2. Derivation of the Sigmoid Derivative

The sigmoid function possesses an elegant mathematical property where its derivative is expressed directly in terms of its output:

$$\begin{aligned}
\sigma(z) &= (1 + e^{-z})^{-1} \\
\frac{d\sigma(z)}{dz} &= -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z}) \\
&= \frac{e^{-z}}{(1 + e^{-z})^2} \\
&= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} \\
&= \frac{1}{1 + e^{-z}} \cdot \left(\frac{1 + e^{-z} - 1}{1 + e^{-z}}\right) \\
&= \sigma(z) \cdot (1 - \sigma(z))
\end{aligned}$$

---

## 3. Comparison: McCulloch-Pitts vs. Sigmoid Neuron

| Property | McCulloch-Pitts Neuron (1943) | Sigmoid Neuron (Logistic Model) |
| :--- | :--- | :--- |
| **Input Domain** | Discrete binary $\{0, 1\}$ | Continuous real numbers $\mathbb{R}^n$ |
| **Weights** | Hard-coded discrete ($+1, -1$) | Continuous learnable parameters $w_i \in \mathbb{R}$ |
| **Threshold / Bias**| Fixed threshold $\theta$ | Learnable additive bias $b \in \mathbb{R}$ |
| **Activation Function** | Heaviside discontinuous step function | Smooth, continuous S-curve (logistic) |
| **Output Range** | Binary $\{0, 1\}$ | Continuous probability $(0, 1)$ |
| **Differentiability** | **Non-differentiable** ($\frac{df}{dz} = 0$ everywhere, undefined at $\theta$) | **Continuously differentiable** ($\sigma'(z) = \sigma(z)(1 - \sigma(z))$) |
| **Learning Capability**| **No learning mechanism** | **Gradient descent backpropagation enabled** |

---

## 4. Why Differentiability Matters for Deep Learning

1. **The Credit Assignment Problem:**  
   Learning requires answering: *"If I tweak weight $w_i$ by a tiny amount $\Delta w_i$, how much will the loss $\mathcal{L}$ increase or decrease?"*
2. **Failure of the Step Function:**  
   For the M-P step function, the derivative $\frac{\partial \mathcal{L}}{\partial w_i} = 0$ almost everywhere. Gradient descent updates:
   $$w_i \leftarrow w_i - \eta \frac{\partial \mathcal{L}}{\partial w_i} = w_i - \eta (0) = w_i$$
   The neuron cannot learn because the gradient provides zero directional guidance.
3. **Success of the Sigmoid:**  
   Because $\sigma'(z) = \sigma(z)(1 - \sigma(z)) > 0$ everywhere, any change in $w_i$ causes a continuous, measurable change in output $\hat{y}$ and loss $\mathcal{L}$, providing a non-zero gradient vector that points downhill toward minimum loss.

---

## 5. Empirical Training Demonstration (Industrial Defect Classification)

A single sigmoid neuron was trained from scratch using Binary Cross-Entropy Loss to classify steel surface conditions (Normal vs. Scratch Defect):
- **Initial State:** Loss = $0.68732$, Accuracy = $62.5\%$, $P(\text{Defect}) \approx 50\%$ across all samples.
- **Converged State (100 Epochs):** Loss = $0.05178$, Accuracy = **$100.0\%$**.
  - Normal samples: $P(\text{Defect}) = 2.63\% \dots 11.63\%$ (Confidently classified Normal).
  - Defect samples: $P(\text{Defect}) = 93.16\% \dots 98.25\%$ (Confidently classified Defect).
- **Gradient Verification:** Finite-difference gradient checking achieved a relative error of **$4.30 \times 10^{-11} \ll 10^{-7}$**.
- **Visualization:** [`docs/figures/sigmoid_characteristics.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/sigmoid_characteristics.png).
