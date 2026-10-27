"""
Sigmoid Neuron Implementation from Scratch
===========================================
First-principles implementation of a continuous, differentiable Sigmoid Neuron.

Mathematical Formulation:
  1. Linear combination: z = sum(w_i * x_i) + b = w^T x + b
  2. Sigmoid activation:   sigma(z) = 1 / (1 + exp(-z))
  3. Sigmoid derivative:   d(sigma)/dz = sigma(z) * (1 - sigma(z))
  4. Binary Cross-Entropy: L = - [y * log(y_hat) + (1 - y) * log(1 - y_hat)]
  5. Gradient updates:     dL/dw = (y_hat - y) * x,  dL/db = (y_hat - y)
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class SigmoidNeuron:
    def __init__(self, num_inputs, seed=42):
        """
        Initializes weights and bias with small values.
        """
        np.random.seed(seed)
        self.weights = np.random.randn(num_inputs) * 0.1
        self.bias = 0.0

    @staticmethod
    def sigmoid(z):
        """
        Numerically stable sigmoid function.
        Prevents overflow for large negative values of z.
        """
        # For z >= 0: 1 / (1 + exp(-z))
        # For z < 0:  exp(z) / (1 + exp(z))
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    @staticmethod
    def sigmoid_derivative(z):
        """
        Computes analytical derivative: d(sigma)/dz = sigma(z) * (1 - sigma(z))
        """
        s = SigmoidNeuron.sigmoid(z)
        return s * (1.0 - s)

    def forward(self, X):
        """
        Computes forward pass:
          z = X @ w + b
          y_hat = sigmoid(z)
        """
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)
        return y_hat, z

    def compute_loss(self, y_hat, y):
        """
        Computes Binary Cross-Entropy (BCE) Loss with clipping for numerical stability.
        """
        eps = 1e-15
        y_hat_clipped = np.clip(y_hat, eps, 1.0 - eps)
        loss = -np.mean(y * np.log(y_hat_clipped) + (1.0 - y) * np.log(1.0 - y_hat_clipped))
        return loss

    def compute_gradients(self, X, y_hat, y):
        """
        Computes analytical gradients for weights and bias via the chain rule:
          dL/dw = (1/N) * X^T (y_hat - y)
          dL/db = (1/N) * sum(y_hat - y)
        """
        N = len(y)
        error = y_hat - y
        grad_w = np.dot(X.T, error) / N
        grad_b = np.sum(error) / N
        return grad_w, grad_b

    def fit(self, X, y, lr=0.5, epochs=100, verbose=True):
        """
        Trains the neuron using standard Batch Gradient Descent.
        """
        history = []
        for epoch in range(1, epochs + 1):
            y_hat, _ = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            grad_w, grad_b = self.compute_gradients(X, y_hat, y)

            # Gradient descent parameter update
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b
            history.append(loss)

            if verbose and (epoch % 20 == 0 or epoch == 1):
                acc = np.mean((y_hat >= 0.5) == y) * 100
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.5f} | Accuracy: {acc:5.1f}%")

        return history

def plot_sigmoid_characteristics(figures_dir="docs/figures"):
    """
    Plots the sigmoid activation S-curve and its derivative to illustrate
    smooth differentiability and the gradient saturation phenomenon.
    """
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    z = np.linspace(-7, 7, 300)
    sig = SigmoidNeuron.sigmoid(z)
    sig_deriv = SigmoidNeuron.sigmoid_derivative(z)

    plt.figure(figsize=(9, 4.5))
    plt.plot(z, sig, label=r"Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$", color='#1d3557', linewidth=2.5)
    plt.plot(z, sig_deriv, label=r"Derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$", color='#e63946', linewidth=2.0, linestyle='--')
    
    plt.axvline(0, color='gray', linestyle=':', alpha=0.6)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.6)
    plt.axhline(1.0, color='black', linestyle='-', alpha=0.2)
    plt.axhline(0.0, color='black', linestyle='-', alpha=0.2)

    # Highlight saturation regions
    plt.annotate('Saturation: Gradient -> 0', xy=(5.5, 0.05), xytext=(3.5, 0.3),
                 arrowprops=dict(arrowstyle='->', color='#e63946', lw=1.5), fontsize=9, fontweight='bold', color='#e63946')
    plt.annotate('Saturation: Gradient -> 0', xy=(-5.5, 0.05), xytext=(-6.8, 0.3),
                 arrowprops=dict(arrowstyle='->', color='#e63946', lw=1.5), fontsize=9, fontweight='bold', color='#e63946')
    plt.annotate('Max Gradient = 0.25 at z=0', xy=(0, 0.25), xytext=(-2.5, 0.6),
                 arrowprops=dict(arrowstyle='->', color='#1d3557', lw=1.5), fontsize=9, fontweight='bold', color='#1d3557')

    plt.title("Sigmoid Activation Function & Continuous Derivative", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Input Logit (z)", fontsize=11, fontweight='semibold')
    plt.ylabel("Value", fontsize=11, fontweight='semibold')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='center right', framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_file = fig_path / "sigmoid_characteristics.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved sigmoid characteristics plot to: {out_file}")

def run_demonstrations():
    print("=" * 65)
    print("  SIGMOID NEURON FROM SCRATCH: THEORY & TRAINING DEMO")
    print("=" * 65)

    # Synthetic Industrial Surface Inspection Problem:
    # Feature 1: Surface roughness metric (normalized)
    # Feature 2: High-frequency contrast ratio (normalized)
    # Target: 0 (Normal metal background), 1 (Scratch defect)
    X = np.array([
        [0.1, 0.2],   # Normal
        [0.2, 0.1],   # Normal
        [0.3, 0.3],   # Normal
        [0.1, 0.4],   # Normal
        [0.8, 0.7],   # Scratch Defect
        [0.9, 0.8],   # Scratch Defect
        [0.7, 0.9],   # Scratch Defect
        [0.85, 0.95]  # Scratch Defect
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    neuron = SigmoidNeuron(num_inputs=2, seed=42)
    print(f"Initial Weights: {neuron.weights.round(4)} | Initial Bias: {neuron.bias:.4f}")

    # Forward pass prior to training
    y_hat_init, _ = neuron.forward(X)
    print("\nInitial Untrained Predictions (Probabilities):")
    for idx, prob in enumerate(y_hat_init):
        print(f"  Sample #{idx} (Target: {y[idx]}) -> P(Defect): {prob*100:5.2f}%")

    # Train using Gradient Descent
    print("\nTraining Sigmoid Neuron via Gradient Descent (BCE Loss)...")
    history = neuron.fit(X, y, lr=2.0, epochs=100, verbose=True)

    # Post-training evaluation
    y_hat_final, _ = neuron.forward(X)
    print(f"\nTrained Weights: {neuron.weights.round(4)} | Trained Bias: {neuron.bias:.4f}")
    print("\nTrained Post-Convergence Predictions:")
    for idx, prob in enumerate(y_hat_final):
        pred_label = 1 if prob >= 0.5 else 0
        status = "[CORRECT]" if pred_label == y[idx] else "[ERROR]"
        print(f"  Sample #{idx} (Target: {y[idx]}) -> P(Defect): {prob*100:5.2f}% -> Pred: {pred_label} {status}")

    # Verify Gradient Checking against Numerical Gradient
    analytical_gw, analytical_gb = neuron.compute_gradients(X, y_hat_final, y)
    eps = 1e-5
    
    # Numerical grad for weights
    num_gw = np.zeros_like(neuron.weights)
    for i in range(len(neuron.weights)):
        w_orig = neuron.weights[i]
        
        neuron.weights[i] = w_orig + eps
        y_pos, _ = neuron.forward(X)
        l_pos = neuron.compute_loss(y_pos, y)
        
        neuron.weights[i] = w_orig - eps
        y_neg, _ = neuron.forward(X)
        l_neg = neuron.compute_loss(y_neg, y)
        
        num_gw[i] = (l_pos - l_neg) / (2.0 * eps)
        neuron.weights[i] = w_orig

    rel_error = np.linalg.norm(num_gw - analytical_gw) / (np.linalg.norm(num_gw) + np.linalg.norm(analytical_gw))
    print(f"\nGradient Check Relative Error on Sigmoid Neuron Weights: {rel_error:.2e}")
    assert rel_error < 1e-7, "Sigmoid neuron gradient check failed!"
    print("[PASSED] Analytical gradients verified against finite-difference numerical gradients.")

    # Generate Visualization
    plot_sigmoid_characteristics()

    print("\n" + "=" * 65)
    print("  ALL SIGMOID NEURON DEMONSTRATIONS PASSED [100%]")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_demonstrations()
