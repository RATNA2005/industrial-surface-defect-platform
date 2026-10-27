"""
Optimizer Comparison Sandbox: Batch GD vs SGD vs Momentum vs Nesterov
======================================================================
Controlled comparison of 4 foundational gradient-based optimizers
implemented from scratch in Python + NumPy on an identical non-linear problem.

Optimizers Evaluated:
  1. Batch Gradient Descent (BGD)
  2. Mini-Batch Stochastic Gradient Descent (SGD)
  3. Classical Momentum
  4. Nesterov Accelerated Gradient (NAG)

Controlled Setup:
  - Identical Dataset: Synthetic metallurgical defect feature classification (200 samples)
  - Identical Architecture: MLP (2 inputs -> 8 hidden -> 1 output)
  - Identical Initial Weights: Fixed seed initialization across all runs
  - Consistent Evaluation: Loss and accuracy evaluated epoch-by-epoch
"""

import sys
import copy
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.mlp_scratch import MultilayerPerceptron

def generate_controlled_dataset(num_samples=200, seed=42):
    """
    Generates a controlled non-linear defect classification problem
    with an ill-conditioned ravine structure to test optimizer dynamics.
    """
    np.random.seed(seed)
    
    # Class 0: Normal metal surface
    n0 = num_samples // 2
    x0_1 = np.random.normal(-0.6, 0.25, n0)
    x0_2 = np.random.normal(0.0, 0.6, n0)
    X0 = np.stack([x0_1, x0_2], axis=1)
    y0 = np.zeros((n0, 1))

    # Class 1: Defect surface (crescent/curved defect)
    n1 = num_samples - n0
    t = np.random.uniform(-1.2, 1.2, n1)
    x1_1 = 0.5 * (t ** 2) + np.random.normal(0.0, 0.08, n1)
    x1_2 = t + np.random.normal(0.0, 0.08, n1)
    X1 = np.stack([x1_1, x1_2], axis=1)
    y1 = np.ones((n1, 1))

    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])

    # Shuffle
    perm = np.random.permutation(num_samples)
    return X[perm], y[perm]

# =====================================================================
# 1. Batch Gradient Descent (BGD)
# =====================================================================
def train_batch_gd(model, X, y, lr=0.5, epochs=150):
    loss_history = []
    for epoch in range(1, epochs + 1):
        y_hat = model.forward(X)
        loss = model.compute_loss(y_hat, y)
        grads = model.backward(y)

        # Standard full-batch update
        model.W1 -= lr * grads["dW1"]
        model.b1 -= lr * grads["db1"]
        model.W2 -= lr * grads["dW2"]
        model.b2 -= lr * grads["db2"]

        loss_history.append(loss)
    return loss_history

# =====================================================================
# 2. Mini-Batch Stochastic Gradient Descent (SGD)
# =====================================================================
def train_sgd(model, X, y, lr=0.5, batch_size=16, epochs=150, seed=42):
    np.random.seed(seed)
    N = X.shape[0]
    loss_history = []

    for epoch in range(1, epochs + 1):
        # Shuffle each epoch
        perm = np.random.permutation(N)
        X_shuff = X[perm]
        y_shuff = y[perm]

        for i in range(0, N, batch_size):
            X_batch = X_shuff[i:i + batch_size]
            y_batch = y_shuff[i:i + batch_size]

            model.forward(X_batch)
            grads = model.backward(y_batch)

            model.W1 -= lr * grads["dW1"]
            model.b1 -= lr * grads["db1"]
            model.W2 -= lr * grads["dW2"]
            model.b2 -= lr * grads["db2"]

        # Track full-dataset loss at end of epoch
        full_pred = model.forward(X)
        loss_history.append(model.compute_loss(full_pred, y))

    return loss_history

# =====================================================================
# 3. Classical Momentum
# =====================================================================
def train_momentum(model, X, y, lr=0.5, gamma=0.9, batch_size=16, epochs=150, seed=42):
    np.random.seed(seed)
    N = X.shape[0]
    loss_history = []

    # Initialize velocity buffers
    v_W1 = np.zeros_like(model.W1)
    v_b1 = np.zeros_like(model.b1)
    v_W2 = np.zeros_like(model.W2)
    v_b2 = np.zeros_like(model.b2)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_shuff = X[perm]
        y_shuff = y[perm]

        for i in range(0, N, batch_size):
            X_batch = X_shuff[i:i + batch_size]
            y_batch = y_shuff[i:i + batch_size]

            model.forward(X_batch)
            grads = model.backward(y_batch)

            # Velocity update: v = gamma * v + lr * grad
            v_W1 = gamma * v_W1 + lr * grads["dW1"]
            v_b1 = gamma * v_b1 + lr * grads["db1"]
            v_W2 = gamma * v_W2 + lr * grads["dW2"]
            v_b2 = gamma * v_b2 + lr * grads["db2"]

            # Parameter update: theta = theta - v
            model.W1 -= v_W1
            model.b1 -= v_b1
            model.W2 -= v_W2
            model.b2 -= v_b2

        full_pred = model.forward(X)
        loss_history.append(model.compute_loss(full_pred, y))

    return loss_history

# =====================================================================
# 4. Nesterov Accelerated Gradient (NAG)
# =====================================================================
def train_nesterov(model, X, y, lr=0.5, gamma=0.9, batch_size=16, epochs=150, seed=42):
    np.random.seed(seed)
    N = X.shape[0]
    loss_history = []

    # Initialize velocity buffers
    v_W1 = np.zeros_like(model.W1)
    v_b1 = np.zeros_like(model.b1)
    v_W2 = np.zeros_like(model.W2)
    v_b2 = np.zeros_like(model.b2)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_shuff = X[perm]
        y_shuff = y[perm]

        for i in range(0, N, batch_size):
            X_batch = X_shuff[i:i + batch_size]
            y_batch = y_shuff[i:i + batch_size]

            # Step 1: Look-ahead step (theta_lookahead = theta - gamma * v)
            model.W1 -= gamma * v_W1
            model.b1 -= gamma * v_b1
            model.W2 -= gamma * v_W2
            model.b2 -= gamma * v_b2

            # Step 2: Evaluate gradients at look-ahead position
            model.forward(X_batch)
            grads_lookahead = model.backward(y_batch)

            # Step 3: Revert lookahead and apply full Nesterov velocity update
            v_W1 = gamma * v_W1 + lr * grads_lookahead["dW1"]
            v_b1 = gamma * v_b1 + lr * grads_lookahead["db1"]
            v_W2 = gamma * v_W2 + lr * grads_lookahead["dW2"]
            v_b2 = gamma * v_b2 + lr * grads_lookahead["db2"]

            model.W1 = model.W1 + gamma * v_W1 - v_W1
            model.b1 = model.b1 + gamma * v_b1 - v_b1
            model.W2 = model.W2 + gamma * v_W2 - v_W2
            model.b2 = model.b2 + gamma * v_b2 - v_b2

        full_pred = model.forward(X)
        loss_history.append(model.compute_loss(full_pred, y))

    return loss_history

def run_optimizer_experiment():
    print("=" * 70)
    print("  CONTROLLED OPTIMIZER COMPARISON (STEP 13)")
    print("=" * 70)

    X, y = generate_controlled_dataset(num_samples=200, seed=42)
    print(f"Dataset: 200 synthetic surface inspection samples (2 features -> Binary Target)")
    print(f"Model: MLP Architecture: 2 Inputs -> 8 Hidden Units -> 1 Output (Sigmoid)")

    # Baseline Model Template (Seed 42)
    base_model = MultilayerPerceptron(input_dim=2, hidden_dim=8, output_dim=1, seed=42)

    epochs = 120
    learning_rate = 0.4
    batch_size = 16
    gamma = 0.85

    optimizers = {
        "Batch GD": (train_batch_gd, {"lr": learning_rate, "epochs": epochs}),
        "Mini-batch SGD": (train_sgd, {"lr": learning_rate, "batch_size": batch_size, "epochs": epochs}),
        "Momentum": (train_momentum, {"lr": learning_rate, "gamma": gamma, "batch_size": batch_size, "epochs": epochs}),
        "Nesterov (NAG)": (train_nesterov, {"lr": learning_rate, "gamma": gamma, "batch_size": batch_size, "epochs": epochs})
    }

    results = {}

    print(f"\nRunning 4 Optimizers for {epochs} Epochs under Identical Initialization:\n")

    for opt_name, (train_fn, kwargs) in optimizers.items():
        # Exact deep copy of baseline initialized weights
        model_instance = copy.deepcopy(base_model)
        
        loss_hist = train_fn(model_instance, X, y, **kwargs)
        
        # Final accuracy
        final_preds = model_instance.forward(X)
        final_acc = float(np.mean((final_preds >= 0.5) == y) * 100)
        final_loss = float(loss_hist[-1])
        initial_loss = float(loss_hist[0])

        # Epochs to reach loss < 0.1
        epochs_to_01 = next((i + 1 for i, l in enumerate(loss_hist) if l < 0.1), None)

        results[opt_name] = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "final_accuracy": final_acc,
            "loss_history": [float(l) for l in loss_hist],
            "epochs_to_threshold_0_1": epochs_to_01
        }

        ep_str = f"{epochs_to_01:3d}" if epochs_to_01 else "Did not reach"
        print(f"  {opt_name:16s} | Init Loss: {initial_loss:.4f} | Final Loss: {final_loss:.5f} | Acc: {final_acc:5.1f}% | Epochs to Loss < 0.1: {ep_str}")

    # Plot Loss vs Epoch Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "Batch GD": "#457b9d",
        "Mini-batch SGD": "#2a9d8f",
        "Momentum": "#e76f51",
        "Nesterov (NAG)": "#e63946"
    }

    # Linear scale
    for opt_name in optimizers:
        axes[0].plot(results[opt_name]["loss_history"], label=opt_name, color=colors[opt_name], linewidth=2.0)
    axes[0].set_title("Training Loss vs. Epoch (Linear Scale)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    axes[0].set_ylabel("Binary Cross-Entropy Loss", fontsize=10, fontweight='semibold')
    axes[0].legend(loc='upper right', framealpha=0.9, fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Logarithmic scale
    for opt_name in optimizers:
        axes[1].semilogy(results[opt_name]["loss_history"], label=opt_name, color=colors[opt_name], linewidth=2.0)
    axes[1].set_title("Training Loss vs. Epoch (Logarithmic Scale)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    axes[1].set_ylabel("Log Loss (log10)", fontsize=10, fontweight='semibold')
    axes[1].legend(loc='upper right', framealpha=0.9, fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Controlled Optimizer Comparison: BGD vs. SGD vs. Momentum vs. Nesterov", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_fig = Path("docs/figures/optimizer_comparison.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"\nSaved optimizer comparison curves to: {out_fig}")

    # Save Results JSON
    summary_path = Path("docs/optimizer_comparison_results.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Exclude giant history from JSON for cleanliness
        clean_summary = {
            k: {sk: sv for sk, sv in v.items() if sk != "loss_history"}
            for k, v in results.items()
        }
        json.dump(clean_summary, f, indent=2)
    print(f"Saved machine-readable summary to: {summary_path}")

    return results

if __name__ == "__main__":
    run_optimizer_experiment()
