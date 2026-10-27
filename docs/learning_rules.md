# Educational Learning-Rule Demonstrations

**Implementation Module:** [`src/learning_rules.py`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/src/learning_rules.py)  
**Figure:** [`docs/figures/learning_rules_summary.png`](file:///c:/Users/Acer/OneDrive/Desktop/dl_project/docs/figures/learning_rules_summary.png)  
**Scope:** Conceptual educational demonstrations of 5 classical learning paradigms in Python + NumPy.

---

## 1. Classical Hebbian Learning (Donald Hebb, 1949)

* **Core Idea:** *"Neurons that fire together, wire together."* Synaptic connection strength increases when pre-synaptic and post-synaptic neurons are simultaneously active.
* **Mathematical Update Rule:**
  $$\Delta w_{ij} = \eta \cdot x_i \cdot y_j, \quad w_{ij}^{(t+1)} = w_{ij}^{(t)} + \Delta w_{ij}$$
* **Small Example:** Repeated exposure to paired metallurgical features (e.g., surface scratch present $+1$ and brightness deficit present $+1$).
* **Empirical Result:** Weights strengthened from $[0.0, 0.0] \to [0.8, 0.8]$ in 4 steps.
* **Conceptual Significance & Limitation:** Shows how correlation alone drives association without supervision. However, without a decay term, weights grow unbounded ($\to \infty$) if patterns repeat indefinitely.

---

## 2. Modified Hebbian Learning: Oja's Rule (Erkki Oja, 1982)

* **Core Idea:** Solves the stability flaw of Hebbian learning by introducing a weight decay term proportional to the squared output $y^2$, mathematically constraining weights to unit length ($\|\mathbf{w}\|_2 \to 1$).
* **Mathematical Update Rule:**
  $$\Delta \mathbf{w} = \eta \cdot y \cdot (\mathbf{x} - y \mathbf{w}), \quad y = \mathbf{w}^T \mathbf{x}$$
* **Small Example:** Applied to a stream of 300 correlated 2D surface defect texture vectors.
* **Empirical Result:**
  - Initial random weights: $[0.9701, -0.2425]$.
  - Final converged weights: $[0.7238, 0.6912]$ with norm $\|\mathbf{w}\|_2 = 1.0008$.
  - True 1st Principal Component (PCA): $[0.7106, 0.7036]$ (Difference: $0.0180$).
* **Conceptual Significance:** Proves that a single neuron equipped with Oja's rule acts as an **online, unsupervised Principal Component Analyzer (PCA)**, extracting the dominant axis of defect feature variance.

---

## 3. Competitive Learning: Winner-Take-All Clustering (Teuvo Kohonen, 1982)

* **Core Idea:** Neurons compete for the right to respond to input patterns. Only the winning neuron (the one with the closest weight vector to the input) is updated toward the input vector.
* **Mathematical Update Rule:**
  $$\text{Winner } c = \arg\min_j \|\mathbf{x} - \mathbf{w}_j\|_2$$
  $$\mathbf{w}_c^{(t+1)} = \mathbf{w}_c^{(t)} + \eta (\mathbf{x} - \mathbf{w}_c^{(t)}), \quad \mathbf{w}_j^{(t+1)} = \mathbf{w}_j^{(t)} \quad (\forall j \neq c)$$
* **Small Example:** Unsupervised clustering of 80 synthetic defect vectors belonging to two clusters: fine scratches (center $[0.2, 0.8]$) and rolled-in scale (center $[0.8, 0.2]$).
* **Empirical Result:** Prototypes initialized at $[0.5, 0.5]$ and $[0.4, 0.6]$ converged autonomously to $[0.178, 0.807]$ and $[0.776, 0.203]$ (distances $< 0.025$ to ground-truth centers).
* **Conceptual Significance:** Demonstrates prototype vector quantization and self-organizing maps (SOM) for unsupervised industrial defect clustering without human labels.

---

## 4. Error-Correction Learning: Widrow-Hoff Delta Rule (1960)

* **Core Idea:** Supervised learning where parameter adjustments are driven directly by the discrepancy (error $e = d - y$) between desired target $d$ and linear model prediction $y$.
* **Mathematical Update Rule:**
  $$e = d - \mathbf{w}^T \mathbf{x}, \quad \Delta \mathbf{w} = \eta \cdot e \cdot \mathbf{x}$$
* **Small Example:** Fitting a linear defect severity predictor ($d = 2.0 x_1 + 3.0 x_2$).
* **Empirical Result:** Weights initialized at $[0.0, 0.0]$ converged to $[2.0059, 2.9954]$ with absolute parameter error $< 0.0075$.
* **Conceptual Significance:** Forms the direct mathematical foundation of the Adaline network, least mean squares (LMS) filter, and modern gradient descent backpropagation in multilayer networks.

---

## 5. Boltzmann Learning / Energy-Based Model (Hopfield & Hinton, 1985)

* **Core Idea:** Recurrent symmetric network where the state of the system is governed by a scalar **Lyapunov Energy Function**. The network acts as an associative content-addressable memory:
  $$E(\mathbf{s}) = -\frac{1}{2} \sum_{i} \sum_{j} w_{ij} s_i s_j$$
  where $s_i \in \{+1, -1\}$.
* **State Transition Dynamics (Glauber / Gibbs Relaxation):**
  $$s_i \leftarrow \operatorname{sign}\left(\sum_j w_{ij} s_j\right)$$
* **Small Example:** Stored a 4-bit clean linear scratch defect prototype: $\mathbf{s}^* = [1, -1, 1, -1]$ (Energy $E = -6.0$, global minimum). Then presented a corrupted probe vector with a flipped bit: $\mathbf{s}_{\text{noisy}} = [1, -1, -1, -1]$ (Energy $E = 0.0$).
* **Empirical Result:** Over asynchronous relaxation, the system descended down the energy landscape, reaching the attractor basin at $E = -6.0$ and completely restoring the clean pattern $[1, -1, 1, -1]$.
* **Conceptual Significance:** Demonstrates associative memory, defect pattern reconstruction from corrupted inputs, and sets the theoretical foundation for Boltzmann Machines and modern Energy-Based Models (EBMs).
