# Numerical Gradient Checking: Theory, Methodology & Verification

**Implementation Module:** [`src/gradient_checking.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/gradient_checking.py)  
**Verification Result:** Relative Error $= 2.575 \times 10^{-11} \ll 10^{-7}$ (100% Passed across all layers)

---

## 1. What Gradient Checking Does

Gradient checking is a rigorous numerical verification procedure that tests whether the analytical gradients computed by backpropagation:
$$\nabla_{\theta} \mathcal{L}_{\text{analytical}} = \left[ \frac{\partial \mathcal{L}}{\partial W_1}, \frac{\partial \mathcal{L}}{\partial \mathbf{b}_1}, \frac{\partial \mathcal{L}}{\partial W_2}, \frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} \right]$$
match the true mathematical definition of a derivative computed via finite differences.

Using two-sided central finite differences with a small perturbation $\epsilon = 10^{-5}$:
$$\frac{\partial \mathcal{L}}{\partial \theta_i}_{\text{numerical}} \approx \frac{\mathcal{L}(\theta + \epsilon \mathbf{e}_i) - \mathcal{L}(\theta - \epsilon \mathbf{e}_i)}{2\epsilon}$$
where $\mathbf{e}_i$ is a unit basis vector with $1$ at parameter index $i$ and $0$ elsewhere.

---

## 2. The Relative Error Metric

Because gradient magnitudes vary widely across layers, raw absolute differences $|\nabla_{\text{analyt}} - \nabla_{\text{num}}|$ are misleading. We evaluate the **normalized, scale-invariant relative error**:

$$\text{Relative Error} = \frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_2}{\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{numerical}}\|_2}$$

### Standard Industry Interpretation Thresholds:
- $\text{Relative Error} < 10^{-7}$: **Verified Correct.** Analytical gradients are mathematically exact within floating-point rounding precision.
- $10^{-7} \le \text{Relative Error} < 10^{-4}$: **Suspicious.** Possible precision loss, $\epsilon$ too large or too small.
- $\text{Relative Error} \ge 10^{-4}$: **Bug in Implementation.** Likely index error, missing transpose, or faulty chain rule term.

---

## 3. Empirical Verification Results

Evaluated on an MLP with 21 scalar parameters across all layers:

| Layer Parameter | Parameter Count | Analytical vs. Numerical Relative Error | Verification Status |
| :--- | :---: | :---: | :---: |
| **Layer 1 Weights ($W_1$)** | 12 | **$1.083 \times 10^{-10}$** | `[PASSED]` |
| **Layer 1 Biases ($\mathbf{b}_1$)** | 4 | **$7.817 \times 10^{-11}$** | `[PASSED]` |
| **Layer 2 Weights ($W_2$)** | 4 | **$6.989 \times 10^{-12}$** | `[PASSED]` |
| **Layer 2 Biases ($\mathbf{b}_2$)** | 1 | **$9.838 \times 10^{-12}$** | `[PASSED]` |
| **Overall Full Network ($\theta$)** | **21** | **$2.575 \times 10^{-11}$** | `[PASSED]` |

---

## 4. Negative Control Test: Intentional Bug Injection

To prove that our gradient checker is sensitive to actual backpropagation bugs, we simulated a classic mistake: forgetting to multiply by the local hidden derivative $\sigma'(a_1)$ in the chain rule:
- **Buggy Error Formulation:** $\delta_1 = \delta_2 W_2^T$ *(omitted $\odot [A_1 \odot (1 - A_1)]$)*.
- **Result:** The relative error immediately spiked from $2.575 \times 10^{-11}$ to **$0.4137$ ($41.4\%$ error)**.
- **Significance:** Demonstrates that the gradient check instantly catches subtle backprop bugs that would otherwise cause models to learn poorly or plateau prematurely.

---

## 5. Why Gradient Checking is Used for Verification, NOT Training

A common viva question is: *"If numerical gradients are so simple and require no calculus, why don't we use them to train neural networks?"*

1. **Computational Complexity:**
   - **Backpropagation:** Computes gradients for **all $P$ parameters simultaneously** in a single backward pass ($O(P)$ complexity).
   - **Numerical Finite Differences:** Requires evaluating the loss function twice for each individual parameter ($2P$ forward passes). For $P$ parameters, total cost is $O(P^2)$.
2. **Real-World Scaling:**
   - In our small MLP with 21 parameters, numerical gradients require $2 \times 21 = 42$ forward passes per step (manageable for checking).
   - In our baseline defect CNN with $\sim 100,000$ parameters, one numerical gradient step would require **$200,000$ forward passes per batch**. A single training epoch would take days instead of seconds.
3. **Conclusion:** Numerical gradient checking is an offline debugging and validation tool used once to prove mathematical correctness, after which analytical backpropagation is used for high-speed training.
