# McCulloch-Pitts (M-P) Neuron: Mathematical Formulation & Implementation

**Implementation Module:** [`src/mcculloch_pitts.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/mcculloch_pitts.py)  
**Historical Context:** Proposed by Warren McCulloch and Walter Pitts in 1943 as the first mathematical abstraction of a biological neuron.

---

## 1. Mathematical Architecture

```
Inputs (x_1, x_2, ..., x_n)
       ↓
Weights (w_1, w_2, ..., w_n)
       ↓
Summation: z = Σ (w_i * x_i)
       ↓
Thresholding: f(z) = 1 if z >= θ else 0
       ↓
Binary Output: y ∈ {0, 1}
```

### Definitions:
1. **Inputs ($x_i$):** Binary signals representing presence or absence of a feature: $x_i \in \{0, 1\}$.
2. **Weights ($w_i$):** Synaptic connection strengths:
   - Excitatory inputs: $w_i = +1$ (increases the internal potential).
   - Inhibitory inputs: $w_i = -\infty$ (absolute veto) or $w_i = -1$ (relative inhibition).
3. **Weighted Sum ($z$):** Linear combination of inputs:
   $$z = \sum_{i=1}^n w_i x_i$$
4. **Threshold ($\theta$):** Predetermined firing threshold required to activate the neuron.
5. **Step Activation Function ($f(z)$):** Discontinuous Heaviside step function:
   $$y = f(z) = \begin{cases} 1 & \text{if } z \ge \theta \\ 0 & \text{if } z < \theta \end{cases}$$
6. **Output ($y$):** Discrete binary decision: $y \in \{0, 1\}$.

---

## 2. Logic Gate Demonstrations

### 2.1 AND Gate ($w_1=1, w_2=1, \theta=2$)
Fires only when **both** inputs are active ($z \ge 2$):

| $x_1$ | $x_2$ | Weighted Sum $z = x_1 + x_2$ | Condition ($z \ge 2$) | Output $y$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | False | **0** |
| 0 | 1 | 1 | False | **0** |
| 1 | 0 | 1 | False | **0** |
| 1 | 1 | 2 | True | **1** |

### 2.2 OR Gate ($w_1=1, w_2=1, \theta=1$)
Fires when **at least one** input is active ($z \ge 1$):

| $x_1$ | $x_2$ | Weighted Sum $z = x_1 + x_2$ | Condition ($z \ge 1$) | Output $y$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | False | **0** |
| 0 | 1 | 1 | True | **1** |
| 1 | 0 | 1 | True | **1** |
| 1 | 1 | 2 | True | **1** |

### 2.3 AND-NOT (Inhibition) Gate ($w_1=+1, w_2=-1, \theta=1$)
Demonstrates inhibition: $x_1$ excites the neuron, but $x_2$ actively inhibits it:

| $x_1$ | $x_2$ | Weighted Sum $z = x_1 - x_2$ | Condition ($z \ge 1$) | Output $y$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | False | **0** |
| 0 | 1 | -1 | False | **0** |
| 1 | 0 | 1 | True | **1** |
| 1 | 1 | 0 | False | **0** |

---

## 3. Industrial Surface Defect Decision Analogy

In manufacturing quality inspection, two optical sensors inspect moving steel strips:
- $x_1$: Specular reflection anomaly sensor (detects scratches).
- $x_2$: Dark-field scatter anomaly sensor (detects rolled-in scale).

```
                      ┌──────────────────────┐
                      │ Sensor 1 (x_1 ∈ {0,1})│
                      └──────────┬───────────┘
                                 │ w_1
                                 ▼
                             [Sum: z] ──► [Threshold θ] ──► Manufacturing Action
                                 ▲
                                 │ w_2
                      ┌──────────┴───────────┐
                      │ Sensor 2 (x_2 ∈ {0,1})│
                      └──────────────────────┘
```

1. **Conservative Inspection Trigger (OR Gate, $\theta=1$):**  
   If *either* sensor detects an irregularity, route the coil to `REVIEW`. Prioritizes zero missed defects (high recall).
2. **Automated Line Rejection Trigger (AND Gate, $\theta=2$):**  
   Halt production and `REJECT` the coil only if *both* sensors confirm defect scattering, avoiding costly false alarm shutdowns.

---

## 4. Fundamental Limitations of the M-P Neuron (Why We Need Modern DL)

1. **No Learning Mechanism:**  
   Weights and thresholds must be hand-configured; the neuron cannot learn from data via optimization.
2. **Discontinuous & Non-Differentiable:**  
   The step function has derivative $\frac{df}{dz} = 0$ everywhere (and undefined at $z = \theta$). Gradient descent cannot compute $\frac{\partial \mathcal{L}}{\partial w}$, making backpropagation impossible.
3. **Linear Separability (The XOR Problem):**  
   A single linear threshold cannot separate non-convex decision boundaries like XOR.
4. **Bridge to Step 10 & 11:**  
   These limitations directly motivate the **Sigmoid Neuron (Step 10)** which introduces continuous differentiability, and the **Multilayer Perceptron (Step 11)** which solves non-linear problems.
