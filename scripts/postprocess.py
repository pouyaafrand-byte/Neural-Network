import json
import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_columns(scaler_stats_path):
    with open(scaler_stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats["output_columns"]


def load_predictions(predictions_path, output_cols):
    df = pd.read_csv(predictions_path)
    y_true = np.column_stack([df[f"{col}_true"].to_numpy() for col in output_cols])
    y_pred = np.column_stack([df[f"{col}_pred"].to_numpy() for col in output_cols])
    return y_true, y_pred


def evaluate_predictions(y_true, y_pred, output_cols):
    rows = []
    for i, col in enumerate(output_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append(
            {
                "output": col,
                "mse": float(mean_squared_error(yt, yp)),
                "mae": float(mean_absolute_error(yt, yp)),
                "r2": float(r2_score(yt, yp)),
            }
        )
    agg = {
        "output": "aggregate",
        "mse": float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))),
        "mae": float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),
        "r2": float(r2_score(y_true.reshape(-1), y_pred.reshape(-1))),
    }
    rows.append(agg)
    return pd.DataFrame(rows)


def plot_identity(y_true, y_pred, output_cols, title, out_path):
    n_outputs = len(output_cols)
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(11, 11))
    axes = axes.ravel()

    for i, col in enumerate(output_cols):
        ax = axes[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        lim_min = min(float(yt.min()), float(yp.min()))
        lim_max = max(float(yt.max()), float(yp.max()))
        r2 = r2_score(yt, yp)
        ax.scatter(yt, yp, s=10, alpha=0.70, color="#1f77b4", edgecolors="none")
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#d62728", linewidth=1.2)
        ax.set_title(f"{col} (R2={r2:.4f})", fontsize=10)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.grid(alpha=0.25)

    agg_idx = n_outputs
    agg_ax = axes[agg_idx]
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    lim_min = min(float(y_true_flat.min()), float(y_pred_flat.min()))
    lim_max = max(float(y_true_flat.max()), float(y_pred_flat.max()))
    r2_agg = r2_score(y_true_flat, y_pred_flat)
    agg_ax.scatter(
        y_true_flat, y_pred_flat, s=8, alpha=0.55, color="#2ca02c", edgecolors="none"
    )
    agg_ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#d62728", linewidth=1.2)
    agg_ax.set_title(f"aggregate (R2={r2_agg:.4f})", fontsize=10)
    agg_ax.set_xlabel("Ground Truth")
    agg_ax.set_ylabel("Prediction")
    agg_ax.grid(alpha=0.25)

    for j in range(agg_idx + 1, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(
        description="Postprocess model predictions and generate identity plots."
    )
    parser.add_argument(
        "--postfix",
        default="_charbonnier",
        help="Artifact postfix to read/write (example: _charbonnier).",
    )
    args = parser.parse_args()

    postfix = args.postfix.strip()
    if postfix and not postfix.startswith("_"):
        postfix = f"_{postfix}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{postfix}_{timestamp}" if postfix else timestamp
    run_tag = run_tag.strip("_")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_root = os.path.join(project_root, "output")
    predictions_dir = os.path.join(output_root, "predictions")
    scalers_dir = os.path.join(output_root, "scalers")
    post_dir = os.path.join(output_root, "postprocessing", run_tag)
    plots_dir = os.path.join(post_dir, "identity_plots")
    metrics_dir = os.path.join(post_dir, "metrics")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    scaler_stats_filename = f"scaler_stats{postfix}.json" if postfix else "scaler_stats.json"
    scaler_stats_path = os.path.join(scalers_dir, scaler_stats_filename)
    if not os.path.exists(scaler_stats_path):
        raise FileNotFoundError(
            f"Missing scaler stats file: {scaler_stats_path}. "
            "Run training with the same postfix first."
        )
    output_cols = load_columns(scaler_stats_path)

    split_files = {
        "train": os.path.join(predictions_dir, f"train_predictions{postfix}.csv"),
        "validation": os.path.join(predictions_dir, f"validation_predictions{postfix}.csv"),
        "test": os.path.join(predictions_dir, f"test_predictions{postfix}.csv"),
    }

    summary = {}
    for split_name, pred_path in split_files.items():
        if not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"Missing predictions file: {pred_path}. Run training first."
            )

        y_true, y_pred = load_predictions(pred_path, output_cols)
        metrics_df = evaluate_predictions(y_true, y_pred, output_cols)
        metrics_path = os.path.join(metrics_dir, f"{split_name}_metrics{postfix}.csv")
        metrics_df.to_csv(metrics_path, index=False)

        plot_path = os.path.join(plots_dir, f"{split_name}_identity_plot{postfix}.png")
        title = f"Identity Plots ({split_name}, postfix={postfix or 'none'})"
        plot_identity(y_true, y_pred, output_cols, title, plot_path)

        agg_row = metrics_df[metrics_df["output"] == "aggregate"].iloc[0]
        summary[split_name] = {
            "aggregate_mse": float(agg_row["mse"]),
            "aggregate_mae": float(agg_row["mae"]),
            "aggregate_r2": float(agg_row["r2"]),
            "metrics_csv": metrics_path,
            "identity_plot": plot_path,
        }

    summary_path = os.path.join(post_dir, f"postprocess_summary{postfix}.json")
    summary["run_info"] = {
        "postfix": postfix,
        "timestamp": timestamp,
        "scaler_stats": scaler_stats_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Postprocessing completed.")
    print(f"Postfix used: {postfix or '(none)'}")
    print(f"Identity plots saved to: {plots_dir}")
    print(f"Metrics saved to: {metrics_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
