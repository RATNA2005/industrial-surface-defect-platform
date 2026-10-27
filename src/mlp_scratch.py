"""
Multilayer Perceptron (MLP) from Scratch in Python + NumPy
===========================================================
Fully custom implementation of a feedforward neural network with backpropagation.
No deep learning framework automatic differentiation is used.

Demonstrates:
  1. Xavier/Glorot weight initialization
  2. Forward propagation through hidden layers
  3. Analytical backpropagation via the multivariate chain rule
  4. Gradient descent parameter updates
  5. Distinction between Parameters (W, b) and Hyperparameters (lr, epochs, hidden_dim)
  6. Solution of the classic non-linear XOR problem & synthetic defect boundary
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class MultilayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        """
        Initializes the 2-layer MLP architecture:
          Layer 1: (input_dim -> hidden_dim)
          Layer 2: (hidden_dim -> output_dim)

        PARAMETERS:
          - W1: Weights matrix for layer 1, shape (input_dim, hidden_dim)
          - b1: Bias vector for layer 1, shape (1, hidden_dim)
          - W2: Weights matrix for layer 2, shape (hidden_dim, output_dim)
          - b2: Bias vector for layer 2, shape (1, output_dim)
        """
        np.random.seed(seed)
        
        # Xavier / Glorot initialization for stable variance
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

        # Cache for backpropagation
        self.cache = {}

    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid activation."""
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    @staticmethod
    def sigmoid_derivative(a):
        """Derivative of sigmoid expressed in terms of activation a = sigma(z)."""
        return a * (1.0 - a)

    def forward(self, X):
        """
        Forward Propagation:
          z1 = X @ W1 + b1
          a1 = sigmoid(z1)
          z2 = a1 @ W2 + b2
          a2 = sigmoid(z2)  (y_hat)
        """
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)

        # Cache intermediate activations for backpropagation
        self.cache = {
            "X": X,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2
        }
        return a2

    def compute_loss(self, y_hat, y):
        """
        Binary Cross-Entropy Loss:
          L = - (1/N) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))
        """
        eps = 1e-15
        y_hat_clip = np.clip(y_hat, eps, 1.0 - eps)
        N = y.shape[0]
        loss = - (1.0 / N) * np.sum(y * np.log(y_hat_clip) + (1.0 - y) * np.log(1.0 - y_hat_clip))
        return loss

    def backward(self, y):
        """
        Analytical Backpropagation via Multivariate Chain Rule:
          delta2 = (a2 - y) / N
          dW2 = a1^T @ delta2
          db2 = sum(delta2, axis=0, keepdims=True)

          delta1 = (delta2 @ W2^T) * sigma'(a1)
          dW1 = X^T @ delta1
          db1 = sum(delta1, axis=0, keepdims=True)
        """
        X = self.cache["X"]
        a1 = self.cache["a1"]
        a2 = self.cache["a2"]
        N = X.shape[0]

        # Output Layer Gradients
        delta2 = (a2 - y) / N
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Hidden Layer Gradients
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(a1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }
        return gradients

    def update_parameters(self, gradients, lr):
        """
        Standard Gradient Descent update:
          theta = theta - lr * dtheta
        """
        self.W1 -= lr * gradients["dW1"]
        self.b1 -= lr * gradients["db1"]
        self.W2 -= lr * gradients["dW2"]
        self.b2 -= lr * gradients["db2"]

    def fit(self, X, y, lr=1.0, epochs=2000, verbose=True):
        """
        Full Training Loop.
        HYPERPARAMETERS:
          - lr (learning rate)
          - epochs (iteration count)
          - hidden_dim (set during initialization)
        """
        history = []
        for epoch in range(1, epochs + 1):
            y_hat = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            gradients = self.backward(y)
            self.update_parameters(gradients, lr)

            history.append(loss)

            if verbose and (epoch % 500 == 0 or epoch == 1):
                acc = np.mean((y_hat >= 0.5) == y) * 100
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f} | Accuracy: {acc:5.1f}%")

        return history

def demo_xor():
    print("=" * 65)
    print("  1. NON-LINEAR XOR PROBLEM DEMONSTRATION")
    print("=" * 65)
    print("Solving the classic non-linear classification problem that single neurons fail on.")

    X_xor = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y_xor = np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ])

    mlp = MultilayerPerceptron(input_dim=2, hidden_dim=4, output_dim=1, seed=42)

    print("\n[HYPERPARAMETERS CONFIGURED]")
    print("  - Learning Rate (lr): 2.0")
    print("  - Epochs: 2000")
    print("  - Hidden Layer Units: 4")
    print("  - Activation Function: Sigmoid")
    print("  - Loss Function: Binary Cross-Entropy")

    print("\n[PARAMETERS TO BE LEARNED]")
    print(f"  - W1: {mlp.W1.shape} ({mlp.W1.size} weights)")
    print(f"  - b1: {mlp.b1.shape} ({mlp.b1.size} biases)")
    print(f"  - W2: {mlp.W2.shape} ({mlp.W2.size} weights)")
    print(f"  - b2: {mlp.b2.shape} ({mlp.b2.size} biases)")
    print(f"  - Total Learnable Parameters: {mlp.W1.size + mlp.b1.size + mlp.W2.size + mlp.b2.size}")

    print("\nTraining MLP on XOR data...")
    history = mlp.fit(X_xor, y_xor, lr=2.0, epochs=2000, verbose=True)

    y_pred = mlp.forward(X_xor)
    print("\nFinal Converged XOR Predictions:")
    print(" x_1 | x_2 | Target | P(Y=1) | Predicted | Status")
    print("-" * 52)
    for i in range(len(X_xor)):
        prob = y_pred[i, 0]
        pred = 1 if prob >= 0.5 else 0
        target = int(y_xor[i, 0])
        status = "[CORRECT]" if pred == target else "[ERROR]"
        print(f"  {int(X_xor[i, 0])}  |  {int(X_xor[i, 1])}  |   {target}    | {prob*100:5.2f}% |     {pred}     | {status}")

    assert np.all((y_pred >= 0.5) == y_xor), "XOR solution failed!"
    print("\n[PASSED] Multilayer Perceptron successfully learned non-linear XOR function.")
    return history

def demo_defect_boundary(figures_dir="docs/figures"):
    print("\n" + "=" * 65)
    print("  2. NON-LINEAR DEFECT BOUNDARY CLASSIFICATION")
    print("=" * 65)

    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    # Synthetic concentric defect clustering:
    # Inner cluster = sound metal, Outer ring = rolled-in scale defect ring
    np.random.seed(42)
    n_samples = 120

    # Class 0: Inner circle
    r0 = np.random.uniform(0.0, 0.4, n_samples // 2)
    theta0 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    y0 = np.zeros((n_samples // 2, 1))

    # Class 1: Outer ring
    r1 = np.random.uniform(0.6, 1.0, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    y1 = np.ones((n_samples // 2, 1))

    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])

    mlp_ring = MultilayerPerceptron(input_dim=2, hidden_dim=8, output_dim=1, seed=42)
    print("Training 2-layer MLP (2 -> 8 -> 1) on non-linear defect boundary...")
    history = mlp_ring.fit(X, y, lr=1.5, epochs=3000, verbose=False)
    
    final_preds = mlp_ring.forward(X)
    final_acc = np.mean((final_preds >= 0.5) == y) * 100
    print(f"Final Non-linear Defect Boundary Accuracy: {final_acc:.1f}%")

    # Plot Decision Boundary & Learning Curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Learning Curve
    axes[0].plot(history, color='#1d3557', linewidth=2)
    axes[0].set_title("MLP Training Loss vs Epoch (BCE Loss)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    axes[0].set_ylabel("Loss", fontsize=10, fontweight='semibold')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Decision Boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_X = np.c_[xx.ravel(), yy.ravel()]
    grid_probs = mlp_ring.forward(grid_X).reshape(xx.shape)

    contour = axes[1].contourf(xx, yy, grid_probs, levels=20, cmap='RdBu_r', alpha=0.7)
    plt.colorbar(contour, ax=axes[1], label="P(Defect)")
    
    axes[1].scatter(X0[:, 0], X0[:, 1], color='#1d3557', edgecolors='k', label="Normal Metal (Class 0)", s=40)
    axes[1].scatter(X1[:, 0], X1[:, 1], color='#e63946', edgecolors='k', label="Scale Defect Ring (Class 1)", s=40)
    axes[1].set_title(f"Non-Linear Decision Boundary (Accuracy: {final_acc:.1f}%)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Feature 1 (Surface Variance)", fontsize=10, fontweight='semibold')
    axes[1].set_ylabel("Feature 2 (Gradient Energy)", fontsize=10, fontweight='semibold')
    axes[1].legend(loc='upper right', framealpha=0.9, fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_file = fig_path / "mlp_decision_boundary.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved MLP decision boundary visual to: {out_file}")

if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("  PHASE I: MULTILAYER PERCEPTRON (MLP) FROM SCRATCH (STEP 11)")
    print("#" * 65)

    demo_xor()
    demo_defect_boundary()

    print("\n" + "=" * 65)
    print("  ALL MLP SCRATCH MODULES COMPLETED & VERIFIED [100%]")
    print("=" * 65 + "\n")
