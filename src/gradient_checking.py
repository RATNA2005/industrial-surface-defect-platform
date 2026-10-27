"""
Comprehensive Numerical Gradient Checking for Multilayer Perceptron
=====================================================================
Rigorous numerical verification of analytical backpropagation gradients
using two-sided central finite differences.

Covers:
  1. Full-parameter vectorization of MLP parameters (W1, b1, W2, b2)
  2. Two-sided finite difference approximation:
       g_num_i = [L(theta + eps * e_i) - L(theta - eps * e_i)] / (2 * eps)
  3. Relative error assessment:
       rel_error = ||g_analyt - g_num|| / (||g_analyt|| + ||g_num||)
  4. Layer-by-layer verification report
  5. Intentional bug injection test (demonstrating that the checker reliably catches backprop flaws)
"""

import sys
import numpy as np
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.mlp_scratch import MultilayerPerceptron

def parameters_to_vector(mlp):
    """Rolls all MLP parameters into a single 1D vector."""
    return np.concatenate([
        mlp.W1.ravel(),
        mlp.b1.ravel(),
        mlp.W2.ravel(),
        mlp.b2.ravel()
    ])

def vector_to_parameters(vector, mlp):
    """Unrolls a 1D vector back into the MLP parameter shapes."""
    idx = 0
    
    w1_size = mlp.W1.size
    mlp.W1 = vector[idx:idx + w1_size].reshape(mlp.W1.shape)
    idx += w1_size

    b1_size = mlp.b1.size
    mlp.b1 = vector[idx:idx + b1_size].reshape(mlp.b1.shape)
    idx += b1_size

    w2_size = mlp.W2.size
    mlp.W2 = vector[idx:idx + w2_size].reshape(mlp.W2.shape)
    idx += w2_size

    b2_size = mlp.b2.size
    mlp.b2 = vector[idx:idx + b2_size].reshape(mlp.b2.shape)

def gradients_to_vector(gradients):
    """Rolls analytical gradients dict into a single 1D vector."""
    return np.concatenate([
        gradients["dW1"].ravel(),
        gradients["db1"].ravel(),
        gradients["dW2"].ravel(),
        gradients["db2"].ravel()
    ])

def compute_numerical_gradient(mlp, X, y, epsilon=1e-5):
    """
    Computes numerical gradient vector via two-sided finite differences:
      grad_num[i] = (loss(theta + eps * e_i) - loss(theta - eps * e_i)) / (2 * eps)
    """
    theta = parameters_to_vector(mlp)
    num_params = len(theta)
    grad_num = np.zeros(num_params)

    for i in range(num_params):
        # theta + epsilon
        theta_pos = theta.copy()
        theta_pos[i] += epsilon
        vector_to_parameters(theta_pos, mlp)
        y_pos = mlp.forward(X)
        loss_pos = mlp.compute_loss(y_pos, y)

        # theta - epsilon
        theta_neg = theta.copy()
        theta_neg[i] -= epsilon
        vector_to_parameters(theta_neg, mlp)
        y_neg = mlp.forward(X)
        loss_neg = mlp.compute_loss(y_neg, y)

        # Central difference
        grad_num[i] = (loss_pos - loss_neg) / (2.0 * epsilon)

    # Restore original parameters
    vector_to_parameters(theta, mlp)
    return grad_num

def compute_relative_error(g_analytical, g_numerical):
    """Computes scale-invariant relative error metric."""
    numerator = np.linalg.norm(g_analytical - g_numerical)
    denominator = np.linalg.norm(g_analytical) + np.linalg.norm(g_numerical)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def run_gradient_check(input_dim=3, hidden_dim=4, output_dim=1, N=8, epsilon=1e-5, seed=42):
    print("=" * 70)
    print("  NUMERICAL GRADIENT CHECKING ACROSS ALL MLP LAYERS")
    print("=" * 70)

    np.random.seed(seed)
    X = np.random.randn(N, input_dim)
    y = np.random.randint(0, 2, (N, output_dim)).astype(np.float64)

    mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim, seed=seed)

    # 1. Compute Analytical Gradients via Backpropagation
    y_hat = mlp.forward(X)
    loss = mlp.compute_loss(y_hat, y)
    analytical_grads_dict = mlp.backward(y)
    analytical_grads_vec = gradients_to_vector(analytical_grads_dict)

    # 2. Compute Numerical Gradients via Finite Differences
    print(f"Computing finite-difference gradients for {len(analytical_grads_vec)} scalar parameters...")
    print(f"Parameters: W1: {mlp.W1.size}, b1: {mlp.b1.size}, W2: {mlp.W2.size}, b2: {mlp.b2.size} (Epsilon: {epsilon})")
    numerical_grads_vec = compute_numerical_gradient(mlp, X, y, epsilon=epsilon)

    # 3. Layer-by-Layer Relative Error Analysis
    print("\n--- Layer-by-Layer Relative Error Breakdown ---")
    idx = 0
    layer_names = ["Layer 1 Weights (W1)", "Layer 1 Biases  (b1)", "Layer 2 Weights (W2)", "Layer 2 Biases  (b2)"]
    param_sizes = [mlp.W1.size, mlp.b1.size, mlp.W2.size, mlp.b2.size]

    for name, size in zip(layer_names, param_sizes):
        g_a = analytical_grads_vec[idx:idx + size]
        g_n = numerical_grads_vec[idx:idx + size]
        rel_err = compute_relative_error(g_a, g_n)
        status = "[PASSED]" if rel_err < 1e-7 else "[FAILED]"
        print(f"  {name:22s} ({size:2d} params) | Rel Error: {rel_err:.3e}  {status}")
        assert rel_err < 1e-7, f"Gradient check failed for {name}"
        idx += size

    # 4. Overall Model Relative Error
    overall_rel_error = compute_relative_error(analytical_grads_vec, numerical_grads_vec)
    print("-" * 70)
    print(f"  OVERALL MLP RELATIVE ERROR: {overall_rel_error:.3e}")
    print(f"  Threshold for Verification:  1.000e-07")
    assert overall_rel_error < 1e-7, "Overall gradient check failed!"
    print("  [VERIFIED] Analytical backpropagation perfectly matches numerical gradients!\n")

    # 5. Intentional Bug Detection Test
    print("=" * 70)
    print("  NEGATIVE CONTROL TEST: INTENTIONAL BUG INJECTION")
    print("=" * 70)
    print("Simulating a subtle common coding error: forgetting the activation derivative sigma'(a1) in backprop.")
    
    # Intentionally corrupted backprop
    X_cache = mlp.cache["X"]
    a1_cache = mlp.cache["a1"]
    a2_cache = mlp.cache["a2"]
    delta2_bug = (a2_cache - y) / N
    dW2_bug = np.dot(a1_cache.T, delta2_bug)
    db2_bug = np.sum(delta2_bug, axis=0, keepdims=True)

    # BUG: delta1 computed without multiplying by a1 * (1 - a1)
    delta1_bug = np.dot(delta2_bug, mlp.W2.T) # Missing sigmoid_derivative!
    dW1_bug = np.dot(X_cache.T, delta1_bug)
    db1_bug = np.sum(delta1_bug, axis=0, keepdims=True)

    buggy_grads = np.concatenate([dW1_bug.ravel(), db1_bug.ravel(), dW2_bug.ravel(), db2_bug.ravel()])
    bug_rel_error = compute_relative_error(buggy_grads, numerical_grads_vec)
    print(f"Buggy Gradient Relative Error: {bug_rel_error:.3e}")
    assert bug_rel_error > 1e-2, "Bug injection failed to trigger error!"
    print("  [CONFIRMED] Gradient checking instantly detected the buggy gradient implementation (Error > 10^-2)!")
    print("=" * 70 + "\n")

    return overall_rel_error

if __name__ == "__main__":
    run_gradient_check()
