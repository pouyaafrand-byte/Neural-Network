"""
Create thesis-style graphs similar to Khoshbakht et al.:

1) Correlation plots (Predicted vs Ground Truth) for each output and aggregated,
   using the best ensemble model [256, 128].
2) (Optional extension) You already have many other plots; this focuses
   on the Reza-style correlation figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plot_correlation_panels(pred_csv: str = "ensemble_256_128_predictions.csv"):
    """
    Create correlation plots like Reza's Figure 6(b):
    Predicted vs Ground Truth for each xi and aggregated.
    """
    df = pd.read_csv(pred_csv)

    # Extract ground truth and predictions
    y_true = df[[f"y_true_x{i}" for i in range(1, 8)]].values
    y_pred = df[[f"y_pred_x{i}" for i in range(1, 8)]].values

    # Compute R2 per output and aggregated
    r2_per_output = [
        r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
    ]
    r2_agg = r2_score(y_true.reshape(-1), y_pred.reshape(-1))

    # 2x4 grid: 7 outputs + aggregated
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(7):
        ax = axes[i]
        gt = y_true[:, i]
        pr = y_pred[:, i]
        r2 = r2_per_output[i]

        ax.scatter(gt, pr, s=15, c="royalblue", alpha=0.7)
        # Perfect prediction line
        min_val = min(gt.min(), pr.min())
        max_val = max(gt.max(), pr.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)

        ax.set_xlabel("Ground Truth", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"x{i+1} (R² = {r2:.4f})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Aggregated panel
    ax = axes[7]
    gt_all = y_true.reshape(-1)
    pr_all = y_pred.reshape(-1)
    ax.scatter(gt_all, pr_all, s=5, c="darkgreen", alpha=0.6)
    min_val = min(gt_all.min(), pr_all.min())
    max_val = max(gt_all.max(), pr_all.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    ax.set_xlabel("Ground Truth", fontsize=9)
    ax.set_ylabel("Predicted", fontsize=9)
    ax.set_title(f"Aggregated (R² = {r2_agg:.4f})", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "thesis_correlation_plots_ensemble_256_128.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Correlation plots saved to: {out_path}")


def main():
    plot_correlation_panels()


if __name__ == "__main__":
    main()

