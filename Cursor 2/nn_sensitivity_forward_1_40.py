import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
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

    return mse_scaled, mse_unscaled


def readable_text_color(rgb):
    r, g, b = rgb[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.6 else "white"


def plot_heatmap(matrix, layers_list, neurons_list, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 11))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(neurons_list)))
    ax.set_xticklabels(neurons_list, color="#f0f0f0", fontsize=11)
    ax.set_yticks(range(len(layers_list)))
    ax.set_yticklabels(layers_list, color="#f0f0f0", fontsize=8)
    ax.set_xlabel("Neurons per layer", color="#f0f0f0", fontsize=12)
    ax.set_ylabel("Number of layers", color="#f0f0f0", fontsize=12)
    ax.set_title(title, color="#f0f0f0", fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Validation MSE (scaled)", color="#f0f0f0")
    cbar.ax.yaxis.set_tick_params(color="#f0f0f0")
    plt.setp(cbar.ax.get_yticklabels(), color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            norm = (val - vmin) / (vmax - vmin + 1e-12)
            color = readable_text_color(cmap(norm))
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
                path_effects=[
                    patheffects.Stroke(linewidth=2, foreground="#111111"),
                    patheffects.Normal(),
                ],
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "DATA.xlsx")
    out_dir = os.path.join(base_dir, "analysis", "nn_sensitivity_forward_1_40")
    os.makedirs(out_dir, exist_ok=True)

    X, Y, input_cols, output_cols = load_data(data_path)
    (
        X_train,
        X_val,
        _X_test,
        Y_train,
        Y_val,
        _Y_test,
        _x_scaler,
        y_scaler,
    ) = split_and_scale(X, Y)

    layers_list = list(range(1, 41))
    neurons_list = [10, 20, 40, 60, 80]
    activations = ["relu", "tanh", "logistic"]
    alphas = [1e-4]

    results = []
    for act in activations:
        for alpha in alphas:
            heatmap_scaled = np.full((len(layers_list), len(neurons_list)), np.nan)
            for i, layers in enumerate(layers_list):
                for j, neurons in enumerate(neurons_list):
                    mse_scaled, mse_unscaled = train_and_eval(
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

            heatmap_path = os.path.join(out_dir, f"heatmap_clean_{act}.png")
            plot_heatmap(
                heatmap_scaled,
                layers_list,
                neurons_list,
                f"Validation MSE vs Layers/Neurons ({act})",
                heatmap_path,
            )

    results_df = pd.DataFrame(results).sort_values("val_mse_scaled")
    results_path = os.path.join(out_dir, "sensitivity_results.csv")
    results_df.to_csv(results_path, index=False)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"inputs: {input_cols}\n")
        f.write(f"outputs: {output_cols}\n")
        f.write(f"layers: {layers_list}\n")
        f.write(f"neurons: {neurons_list}\n")

    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
