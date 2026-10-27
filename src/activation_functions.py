"""
Activation Functions: Mathematical Comparison & Visualization
=============================================================
Comparative analysis of 4 foundational activation functions and their
derivatives:
  1. Step (Heaviside)
  2. Sigmoid (Logistic)
  3. Hyperbolic Tangent (Tanh)
  4. Rectified Linear Unit (ReLU)
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =====================================================================
# Activation Functions & Analytical Derivatives
# =====================================================================
def step_func(z):
    """Heaviside Step Function."""
    return np.where(z >= 0, 1.0, 0.0)

def step_derivative(z):
    """Derivative of Step (0 everywhere, undefined at 0; treated as 0)."""
    return np.zeros_like(z)

def sigmoid_func(z):
    """Numerically stable Sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )

def sigmoid_derivative(z):
    """Analytical derivative of Sigmoid: sigma(z) * (1 - sigma(z))."""
    s = sigmoid_func(z)
    return s * (1.0 - s)

def tanh_func(z):
    """Hyperbolic Tangent Function."""
    return np.tanh(z)

def tanh_derivative(z):
    """Analytical derivative of Tanh: 1 - tanh^2(z)."""
    t = np.tanh(z)
    return 1.0 - (t ** 2)

def relu_func(z):
    """Rectified Linear Unit."""
    return np.maximum(0.0, z)

def relu_derivative(z):
    """Analytical derivative of ReLU (1 for z > 0, 0 for z <= 0)."""
    return np.where(z > 0, 1.0, 0.0)

# =====================================================================
# Visualization & Comparative Plotting
# =====================================================================
def generate_activation_plots(figures_dir="docs/figures"):
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    z = np.linspace(-4.5, 4.5, 500)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)

    funcs = [
        ("Step (Heaviside)", step_func(z), step_derivative(z), "{0, 1}", "None (Zero everywhere)", "#6c757d"),
        ("Sigmoid (Logistic)", sigmoid_func(z), sigmoid_derivative(z), "(0, 1)", "Max 0.25 at z=0", "#1d3557"),
        ("Tanh", tanh_func(z), tanh_derivative(z), "(-1, 1)", "Max 1.00 at z=0", "#2a9d8f"),
        ("ReLU", relu_func(z), relu_derivative(z), "[0, inf)", "Constant 1.0 (z>0)", "#e63946")
    ]

    for col, (name, y_val, dy_val, out_range, max_grad, color) in enumerate(funcs):
        # Top Row: Function f(z)
        ax_top = axes[0, col]
        ax_top.plot(z, y_val, color=color, linewidth=2.5, label=name)
        ax_top.set_title(f"{name}\nRange: {out_range}", fontsize=11, fontweight='bold')
        ax_top.grid(True, linestyle='--', alpha=0.5)
        ax_top.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax_top.axvline(0, color='gray', linestyle=':', alpha=0.7)
        if col == 0:
            ax_top.set_ylabel("Activation f(z)", fontsize=11, fontweight='semibold')

        # Annotate specific function limits
        if name == "Step (Heaviside)":
            ax_top.set_ylim(-0.2, 1.3)
        elif name == "Sigmoid (Logistic)":
            ax_top.set_ylim(-0.1, 1.1)
            ax_top.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
        elif name == "Tanh":
            ax_top.set_ylim(-1.2, 1.2)
            ax_top.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
            ax_top.axhline(-1.0, color='gray', linestyle=':', alpha=0.4)
        elif name == "ReLU":
            ax_top.set_ylim(-0.5, 4.5)

        # Bottom Row: Derivative f'(z)
        ax_bot = axes[1, col]
        ax_bot.plot(z, dy_val, color=color, linewidth=2.0, linestyle='--', label=f"{name}'")
        ax_bot.set_title(f"Derivative f'(z)\n{max_grad}", fontsize=10, fontweight='semibold')
        ax_bot.grid(True, linestyle='--', alpha=0.5)
        ax_bot.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax_bot.axvline(0, color='gray', linestyle=':', alpha=0.7)
        ax_bot.set_xlabel("Input Logit (z)", fontsize=10, fontweight='semibold')
        if col == 0:
            ax_bot.set_ylabel("Derivative f'(z)", fontsize=11, fontweight='semibold')

        if name == "ReLU":
            ax_bot.set_ylim(-0.1, 1.2)
            ax_bot.annotate('Dying ReLU\n(Grad = 0)', xy=(-2.0, 0.05), xytext=(-3.8, 0.5),
                            arrowprops=dict(arrowstyle='->', color='#e63946', lw=1.5), fontsize=8, fontweight='bold', color='#e63946')
        elif name in ["Sigmoid (Logistic)", "Tanh"]:
            ax_bot.set_ylim(-0.05, 1.1)
            ax_bot.annotate('Vanishing\nGradient', xy=(3.5, 0.05), xytext=(2.2, 0.5),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.2), fontsize=8, fontweight='bold', color=color)

    plt.suptitle("Comparative Analysis of Activation Functions & First-Order Derivatives", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_file = fig_path / "activation_functions_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved activation functions comparison plot to: {out_file}")

# =====================================================================
# Console Verification & Mathematical Summary
# =====================================================================
def run_comparison():
    print("=" * 75)
    print("  ACTIVATION FUNCTIONS: MATHEMATICAL BEHAVIOUR & PROPERTIES")
    print("=" * 75)

    test_points = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    
    print(f"\nEvaluation on Test Points z = {test_points}:\n")
    print("Function   | z = -3.0  | z = -1.0  | z =  0.0  | z = +1.0  | z = +3.0")
    print("-" * 65)

    for name, fn in [("Step", step_func), ("Sigmoid", sigmoid_func), ("Tanh", tanh_func), ("ReLU", relu_func)]:
        vals = [f"{v:+8.4f}" for v in fn(test_points)]
        print(f"{name:10s} | " + " | ".join(vals))

    print("\nDerivatives f'(z):\n")
    print("Function'  | z = -3.0  | z = -1.0  | z =  0.0  | z = +1.0  | z = +3.0")
    print("-" * 65)

    for name, dfn in [("Step'", step_derivative), ("Sigmoid'", sigmoid_derivative), ("Tanh'", tanh_derivative), ("ReLU'", relu_derivative)]:
        dvals = [f"{v:+8.4f}" for v in dfn(test_points)]
        print(f"{name:10s} | " + " | ".join(dvals))

    print("\n" + "=" * 75)
    print("  KEY ARCHITECTURAL TAKEAWAYS FOR OUR INDUSTRIAL INSPECTION CNN")
    print("=" * 75)
    print("1. Hidden Layers: Use ReLU (avoids vanishing gradients during backprop through convolutions).")
    print("2. Output Layer:  Use Softmax (generalization of Sigmoid for 6-class surface defect probabilities).")
    print("=" * 75 + "\n")

    generate_activation_plots()

if __name__ == "__main__":
    run_comparison()
