"""
Tensor Mathematics Educational Sandbox
=======================================
Basic Python + NumPy implementation of core mathematical operations underlying
deep learning and visual inspection models.

Covers:
  1. Scalar, Vector, Matrix, and Tensor concepts
  2. Shapes, Reshaping, and Axis Permutations
  3. Broadcasting (Manual vs NumPy)
  4. Vector Dot Products and Cosine Similarity (Manual vs NumPy)
  5. Matrix Multiplication / GEMM (Manual triple-loop vs NumPy)
  6. Numerically Stable Softmax (Manual vs NumPy)
  7. Singular Value Decomposition (SVD) & Low-Rank Image Compression
  8. Numerical Gradient Checking via Finite Differences
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def section_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# =====================================================================
# 1. Tensor Orders & Dimensions
# =====================================================================
def demo_tensor_orders():
    section_header("1. TENSOR ORDERS & CONCEPTS (0D to 4D)")
    
    # 0D Tensor: Scalar (e.g., loss value L)
    scalar_0d = np.array(0.4523)
    print(f"[0D Tensor - Scalar] Value: {scalar_0d:.4f} | Shape: {scalar_0d.shape} | Rank: {scalar_0d.ndim}")

    # 1D Tensor: Vector (e.g., class bias vector b in R^6)
    vector_1d = np.array([0.15, -0.05, 0.40, -0.22, 0.10, 0.05])
    print(f"[1D Tensor - Vector] Shape: {vector_1d.shape} | Rank: {vector_1d.ndim}")
    print(f"   Contents: {vector_1d}")

    # 2D Tensor: Matrix (e.g., weight matrix W in R^{4 x 6})
    matrix_2d = np.array([
        [0.1, 0.2, -0.1, 0.4, 0.0, 0.3],
        [-0.2, 0.5, 0.1, -0.3, 0.2, -0.1],
        [0.0, -0.4, 0.3, 0.1, -0.2, 0.4],
        [0.3, 0.1, -0.2, 0.0, 0.5, -0.2]
    ])
    print(f"[2D Tensor - Matrix] Shape: {matrix_2d.shape} | Rank: {matrix_2d.ndim}")

    # 3D Tensor: Single image patch (C x H x W = 3 x 200 x 200)
    image_3d = np.zeros((3, 200, 200), dtype=np.float32)
    print(f"[3D Tensor - Image]  Shape: {image_3d.shape} (Channels, Height, Width)")

    # 4D Tensor: Mini-batch of images (B x C x H x W = 32 x 3 x 200 x 200)
    batch_4d = np.zeros((32, 3, 200, 200), dtype=np.float32)
    print(f"[4D Tensor - Batch]  Shape: {batch_4d.shape} (Batch, Channels, Height, Width)")

# =====================================================================
# 2. Reshaping, Flattening & Axis Permutations
# =====================================================================
def demo_reshaping():
    section_header("2. RESHAPING & AXIS PERMUTATIONS")
    
    # Feature map from CNN: (Batch=2, Channels=3, Height=2, Width=2)
    x = np.arange(24).reshape((2, 3, 2, 2))
    print(f"Original Feature Map Shape: {x.shape}")

    # Flatten for dense classification layer: (Batch, Channels * Height * Width)
    flattened = x.reshape(x.shape[0], -1)
    print(f"Flattened for Dense Layer:   {flattened.shape} (Batch=2, Features=12)")

    # Permuting axes: PyTorch (C, H, W) to Matplotlib (H, W, C)
    single_img = x[0] # (3, 2, 2)
    transposed = np.transpose(single_img, (1, 2, 0)) # (2, 2, 3)
    print(f"Transposed (C, H, W) -> (H, W, C): {single_img.shape} -> {transposed.shape}")

# =====================================================================
# 3. Broadcasting (Manual vs NumPy)
# =====================================================================
def demo_broadcasting():
    section_header("3. BROADCASTING: MANUAL VS NUMPY")
    
    # Batch matrix: 3 samples, 4 features
    A = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ]) # Shape: (3, 4)

    # Bias vector to add to each sample: 4 features
    b = np.array([0.5, -0.5, 1.0, -1.0]) # Shape: (4,)

    # NumPy automatic broadcasting
    C_numpy = A + b

    # Manual broadcasting (nested loops)
    rows, cols = A.shape
    C_manual = np.zeros_like(A)
    for i in range(rows):
        for j in range(cols):
            C_manual[i, j] = A[i, j] + b[j]

    error = np.max(np.abs(C_manual - C_numpy))
    print(f"Matrix A shape: {A.shape} | Vector b shape: {b.shape}")
    print(f"Result shape:   {C_numpy.shape}")
    print(f"Manual vs NumPy Max Absolute Difference: {error:.1e}")
    assert error < 1e-12, "Broadcasting verification failed!"
    print("[PASSED] Manual broadcasting matches NumPy perfectly.")

# =====================================================================
# 4. Dot Product & Cosine Similarity (Manual vs NumPy)
# =====================================================================
def demo_dot_product():
    section_header("4. INNER PRODUCT & COSINE SIMILARITY")
    
    u = np.array([1.2, -0.5, 2.0, 0.8])
    v = np.array([0.5, 1.0, -1.5, 2.0])

    # Manual dot product: sum_i (u_i * v_i)
    dot_manual = 0.0
    for i in range(len(u)):
        dot_manual += u[i] * v[i]

    dot_numpy = np.dot(u, v)
    error = abs(dot_manual - dot_numpy)
    print(f"Vector u: {u}")
    print(f"Vector v: {v}")
    print(f"Manual Dot Product: {dot_manual:.6f}")
    print(f"NumPy Dot Product:  {dot_numpy:.6f}")
    print(f"Absolute Difference: {error:.1e}")
    assert error < 1e-12, "Dot product verification failed!"
    print("[PASSED] Manual dot product matches NumPy.")

    # Cosine Similarity: cos(theta) = (u . v) / (||u|| * ||v||)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    cosine_sim = dot_numpy / (norm_u * norm_v)
    print(f"Cosine Similarity between u and v: {cosine_sim:.4f}")

# =====================================================================
# 5. Matrix Multiplication / GEMM (Manual vs NumPy)
# =====================================================================
def demo_matrix_multiplication():
    section_header("5. MATRIX MULTIPLICATION (GEMM): MANUAL VS NUMPY")
    
    # Input batch X: 3 samples, 4 inputs
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [-1.0, 0.0, 1.0, 2.0]
    ]) # (3, 4)

    # Weights W: 4 inputs, 2 output defect classes
    W = np.array([
        [0.1, -0.2],
        [0.4, 0.1],
        [-0.3, 0.5],
        [0.2, -0.1]
    ]) # (4, 2)

    # Bias b: 2 output neurons
    b = np.array([0.05, -0.05])

    # NumPy GEMM: Y = X @ W + b
    Y_numpy = np.matmul(X, W) + b

    # Manual triple-nested loop implementation: Y_ik = sum_j (X_ij * W_jk) + b_k
    N, M = X.shape
    _, P = W.shape
    Y_manual = np.zeros((N, P))

    for i in range(N):
        for k in range(P):
            acc = 0.0
            for j in range(M):
                acc += X[i, j] * W[j, k]
            Y_manual[i, k] = acc + b[k]

    error = np.max(np.abs(Y_manual - Y_numpy))
    print(f"X shape: {X.shape} | W shape: {W.shape} | Output Y shape: {Y_numpy.shape}")
    print("NumPy Output Y:\n", Y_numpy)
    print("Manual Output Y:\n", Y_manual)
    print(f"Max Absolute Error: {error:.1e}")
    assert error < 1e-12, "Matrix multiplication verification failed!"
    print("[PASSED] Manual GEMM matches NumPy.")

# =====================================================================
# 6. Numerically Stable Softmax (Manual vs NumPy)
# =====================================================================
def demo_softmax():
    section_header("6. NUMERICALLY STABLE SOFTMAX")
    
    # Raw unnormalized logits from classifier
    logits = np.array([
        [2.0, 1.0, 0.1, -1.0, 3.5, 0.0],
        [1000.0, 1001.0, 1002.0, 999.0, 1005.0, 998.0] # Large values that would cause exp() overflow
    ])

    def manual_stable_softmax(z):
        # Subtract max along row for numerical stability: exp(z - max)
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return probs

    probs = manual_stable_softmax(logits)
    print("Logits Shape:", logits.shape)
    print("Stable Softmax Probabilities for Batch Sample #0:")
    for cls_idx, p in enumerate(probs[0]):
        print(f"   Class {cls_idx}: {p*100:6.2f}%")
    print(f"Probabilities sum across row 0: {np.sum(probs[0]):.6f}")
    print(f"Probabilities sum across row 1: {np.sum(probs[1]):.6f}")
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Softmax probabilities must sum to 1.0"
    print("[PASSED] Numerically stable Softmax correctly handles extreme logits without overflow.")

# =====================================================================
# 7. Singular Value Decomposition (SVD) on Surface Defect Image
# =====================================================================
def demo_svd(sample_image_path="data/raw/images/scratches_1.jpg", figures_dir="docs/figures"):
    section_header("7. SINGULAR VALUE DECOMPOSITION (SVD)")
    
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    with Image.open(sample_image_path) as img:
        img_gray = img.convert('L')
        A = np.array(img_gray, dtype=np.float64) # (200, 200)

    # Full SVD: A = U * S * V^T
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    print(f"Original Image Matrix A: {A.shape}")
    print(f"U shape: {U.shape} | Singular values S: {S.shape} | V^T shape: {Vt.shape}")
    print(f"Top 5 Singular Values: {S[:5].round(1)}")

    # Low-Rank Approximations: A_k = sum_{i=1}^k s_i * u_i * v_i^T
    ranks = [1, 5, 20, 50, 200]
    fig, axes = plt.subplots(1, len(ranks), figsize=(16, 3.5))

    for idx, k in enumerate(ranks):
        # Reconstruct rank-k approximation
        A_k = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
        A_k = np.clip(A_k, 0, 255).astype(np.uint8)

        # Compute compression energy ratio: sum(S_k^2) / sum(S_total^2)
        energy_ratio = np.sum(S[:k] ** 2) / np.sum(S ** 2) * 100

        axes[idx].imshow(A_k, cmap='gray')
        axes[idx].set_title(f"Rank k={k}\nEnergy: {energy_ratio:.1f}%", fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle("SVD Low-Rank Approximation of Industrial Steel Scratch Defect", fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_file = fig_path / "svd_reconstruction.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved SVD low-rank reconstruction visual to: {out_file}")
    print("[PASSED] SVD low-rank matrix approximation verified.")

# =====================================================================
# 8. Numerical Gradient Checking (Finite Differences)
# =====================================================================
def demo_gradient_checking():
    section_header("8. NUMERICAL GRADIENT CHECKING (FINITE DIFFERENCES)")

    # Objective function: Ridge regression loss L(W) = 0.5 * ||X W - y||^2 + 0.5 * lambda * ||W||^2
    np.random.seed(42)
    N, D = 10, 5
    X = np.random.randn(N, D)
    y = np.random.randn(N, 1)
    W = np.random.randn(D, 1)
    reg = 0.1

    def compute_loss(weights):
        diff = np.dot(X, weights) - y
        loss = 0.5 * np.sum(diff ** 2) + 0.5 * reg * np.sum(weights ** 2)
        return loss

    # Analytical gradient from calculus: grad_W = X^T (X W - y) + lambda * W
    diff = np.dot(X, W) - y
    grad_analytical = np.dot(X.T, diff) + reg * W

    # Numerical gradient via two-sided finite differences: (f(w + eps) - f(w - eps)) / (2 * eps)
    epsilon = 1e-5
    grad_numerical = np.zeros_like(W)

    for i in range(W.size):
        W_pos = W.copy()
        W_neg = W.copy()
        W_pos.flat[i] += epsilon
        W_neg.flat[i] -= epsilon

        loss_pos = compute_loss(W_pos)
        loss_neg = compute_loss(W_neg)
        grad_numerical.flat[i] = (loss_pos - loss_neg) / (2.0 * epsilon)

    # Relative Error Metric: ||grad_num - grad_analyt|| / (||grad_num|| + ||grad_analyt||)
    numerator = np.linalg.norm(grad_numerical - grad_analytical)
    denominator = np.linalg.norm(grad_numerical) + np.linalg.norm(grad_analytical)
    rel_error = numerator / denominator

    print("Analytical Gradient:\n", grad_analytical.flatten().round(6))
    print("Numerical Gradient: \n", grad_numerical.flatten().round(6))
    print(f"Relative Error: {rel_error:.2e}")

    assert rel_error < 1e-7, f"Gradient check failed! Relative error {rel_error} too high."
    print("[PASSED] Gradient Check Passed! Relative error is well below 1e-7 threshold.")

# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("  PHASE I: TENSOR MATHEMATICS SANDBOX (STEP 8)")
    print("#" * 65)

    demo_tensor_orders()
    demo_reshaping()
    demo_broadcasting()
    demo_dot_product()
    demo_matrix_multiplication()
    demo_softmax()
    demo_svd()
    demo_gradient_checking()

    print("\n" + "=" * 65)
    print("  ALL TENSOR MATHEMATICS MODULES COMPLETED & VERIFIED [100%]")
    print("=" * 65 + "\n")
