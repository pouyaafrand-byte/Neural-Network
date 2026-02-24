import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as patheffects
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_excel(path)
    input_cols = ["mz", "q", "fx", "fy"]
    output_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    X = df[input_cols].to_numpy()
    Y = df[output_cols].to_numpy()
    return X, Y, input_cols, output_cols


def split_and_scale(X, Y, seed=42):
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.30, random_state=seed
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.50, random_state=seed
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    Y_train_s = y_scaler.fit_transform(Y_train)
    Y_val_s = y_scaler.transform(Y_val)
    Y_test_s = y_scaler.transform(Y_test)

    return (
        X_train_s,
        X_val_s,
        X_test_s,
        Y_train_s,
        Y_val_s,
        Y_test_s,
        x_scaler,
        y_scaler,
    )


def train_and_eval(
    X_train,
    Y_train,
    X_val,
    Y_val,
    y_scaler,
    layers,
    neurons,
    activation,
    alpha,
    seed=42,
):
    hidden = tuple([neurons] * layers)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver="adam",
        learning_rate_init=0.003,
        batch_size=64,
        early_stopping=True,
        n_iter_no_change=4,
        max_iter=60,
        tol=1e-4,
        alpha=alpha,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, Y_train)
    pred_val = model.predict(X_val)
    mse_scaled = mean_squared_error(Y_val, pred_val)

    pred_val_unscaled = y_scaler.inverse_transform(pred_val)
    y_val_unscaled = y_scaler.inverse_transform(Y_val)
    mse_unscaled = mean_squared_error(y_val_unscaled, pred_val_unscaled)

    return model, mse_scaled, mse_unscaled


def plot_heatmap(matrix, layers_list, neurons_list, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(neurons_list)))
    ax.set_xticklabels(neurons_list, color="#f7f7f7")
    ax.set_yticks(range(len(layers_list)))
    ax.set_yticklabels(layers_list, color="#f7f7f7")
    ax.set_xlabel("Neurons per layer", color="#f7f7f7")
    ax.set_ylabel("Number of layers", color="#f7f7f7")
    ax.set_title(title, color="#f7f7f7")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Validation MSE", color="#f7f7f7")
    cbar.ax.yaxis.set_tick_params(color="#f7f7f7")
    plt.setp(cbar.ax.get_yticklabels(), color="#f7f7f7")
    ax.tick_params(colors="#f7f7f7")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            norm_val = (value - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix) + 1e-12)
            text_color = "black" if norm_val > 0.6 else "white"
            ax.text(
                j,
                i,
                f"{value:.2e}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                path_effects=[
                    patheffects.Stroke(linewidth=2, foreground="#0b0b0b"),
                    patheffects.Normal(),
                ],
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_predictions(y_true, y_pred, output_cols, out_path):
    n_outputs = len(output_cols)
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()

    for i, col in enumerate(output_cols):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], s=8, alpha=0.7)
        lims = [
            min(y_true[:, i].min(), y_pred[:, i].min()),
            max(y_true[:, i].max(), y_pred[:, i].max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1)
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.set_title(f"{col} (R2={r2:.4f})", fontsize=9)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")

    agg_ax = axes[n_outputs]
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    agg_ax.scatter(y_true_flat, y_pred_flat, s=6, alpha=0.6)
    lims = [
        min(y_true_flat.min(), y_pred_flat.min()),
        max(y_true_flat.max(), y_pred_flat.max()),
    ]
    agg_ax.plot(lims, lims, "r--", linewidth=1)
    r2 = r2_score(y_true_flat, y_pred_flat)
    agg_ax.set_title(f"Aggregated (R2={r2:.4f})", fontsize=9)
    agg_ax.set_xlabel("Ground Truth")
    agg_ax.set_ylabel("Predicted")

    for j in range(n_outputs + 1, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle("Prediction vs Ground Truth (Test Set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "DATA.xlsx")
    out_dir = os.path.join(base_dir, "analysis", "nn_sensitivity_mqfxfy")
    os.makedirs(out_dir, exist_ok=True)

    X, Y, input_cols, output_cols = load_data(data_path)
    (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test,
        x_scaler,
        y_scaler,
    ) = split_and_scale(X, Y)

    layers_list = list(range(1, 31))
    neurons_list = [20, 40]
    default_activations = ["relu", "tanh", "logistic"]
    activations_env = os.environ.get("NN_ACTIVATIONS")
    if activations_env:
        activations = [a.strip() for a in activations_env.split(",") if a.strip()]
    else:
        activations = default_activations
    alphas = [1e-4]

    results = []
    best = {"mse": np.inf}

    for act in activations:
        for alpha in alphas:
            print(f"Running activation={act}, alpha={alpha}")
            heatmap_scaled = np.full((len(layers_list), len(neurons_list)), np.nan)
            heatmap_unscaled = np.full((len(layers_list), len(neurons_list)), np.nan)
            for i, layers in enumerate(layers_list):
                print(f"  Layers={layers}")
                for j, neurons in enumerate(neurons_list):
                    model, mse_scaled, mse_unscaled = train_and_eval(
                        X_train,
                        Y_train,
                        X_val,
                        Y_val,
                        y_scaler,
                        layers,
                        neurons,
                        act,
                        alpha,
                    )
                    heatmap_scaled[i, j] = mse_scaled
                    heatmap_unscaled[i, j] = mse_unscaled
                    results.append(
                        {
                            "activation": act,
                            "alpha": alpha,
                            "layers": layers,
                            "neurons": neurons,
                            "val_mse_scaled": mse_scaled,
                            "val_mse_unscaled": mse_unscaled,
                        }
                    )
                    if mse_scaled < best["mse"]:
                        best = {
                            "mse": mse_scaled,
                            "activation": act,
                            "alpha": alpha,
                            "layers": layers,
                            "neurons": neurons,
                            "model": model,
                        }

            alpha_label = f"{alpha:.0e}".replace("e-0", "e-")
            heatmap_path = os.path.join(out_dir, f"heatmap_{act}_a{alpha_label}.png")
            plot_heatmap(
                heatmap_scaled,
                layers_list,
                neurons_list,
                f"Validation Loss Heatmap (scaled, {act}, alpha={alpha_label})",
                heatmap_path,
            )
            heatmap_unscaled_path = os.path.join(
                out_dir, f"heatmap_{act}_a{alpha_label}_unscaled.png"
            )
            plot_heatmap(
                heatmap_unscaled,
                layers_list,
                neurons_list,
                f"Validation Loss Heatmap (unscaled, {act}, alpha={alpha_label})",
                heatmap_unscaled_path,
            )

            results_df = pd.DataFrame(results).sort_values("val_mse_scaled")
            results_path = os.path.join(out_dir, "sensitivity_results.csv")
            results_df.to_csv(results_path, index=False)

    results_df = pd.DataFrame(results).sort_values("val_mse_scaled")
    results_path = os.path.join(out_dir, "sensitivity_results.csv")
    results_df.to_csv(results_path, index=False)

    best_model = best["model"]
    pred_test = best_model.predict(X_test)
    pred_test_unscaled = y_scaler.inverse_transform(pred_test)
    y_test_unscaled = y_scaler.inverse_transform(Y_test)

    plot_path = os.path.join(out_dir, "prediction_scatter.png")
    plot_predictions(y_test_unscaled, pred_test_unscaled, output_cols, plot_path)

    summary = {
        "best_activation": best["activation"],
        "best_alpha": best["alpha"],
        "best_layers": best["layers"],
        "best_neurons": best["neurons"],
        "best_val_mse": best["mse"],
        "test_mse": mean_squared_error(Y_test, pred_test),
    }
    pred_test_unscaled = y_scaler.inverse_transform(pred_test)
    y_test_unscaled = y_scaler.inverse_transform(Y_test)
    mae_unscaled = np.mean(np.abs(pred_test_unscaled - y_test_unscaled), axis=0)
    denom = np.maximum(np.abs(y_test_unscaled), 1e-8)
    mape_unscaled = np.mean(np.abs((pred_test_unscaled - y_test_unscaled) / denom), axis=0) * 100.0
    summary["test_mae_unscaled_mean"] = float(np.mean(mae_unscaled))
    summary["test_mape_unscaled_mean_percent"] = float(np.mean(mape_unscaled))
    summary["test_mae_unscaled_per_output"] = mae_unscaled.tolist()
    summary["test_mape_unscaled_percent_per_output"] = mape_unscaled.tolist()
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("Saved outputs to:", out_dir)
    print("Best model:", summary)


if __name__ == "__main__":
    main()
