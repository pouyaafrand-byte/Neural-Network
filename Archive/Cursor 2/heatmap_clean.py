import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects


def readable_text_color(rgb):
    r, g, b = rgb[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.6 else "white"


def plot_heatmap(matrix, layers, neurons, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    cmap = plt.get_cmap("viridis")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(neurons)))
    ax.set_xticklabels(neurons, color="#f0f0f0", fontsize=11)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, color="#f0f0f0", fontsize=10)
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
                fontsize=9,
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
    in_path = os.path.join(base_dir, "analysis", "nn_sensitivity_mqfxfy", "sensitivity_results.csv")
    out_dir = os.path.join(base_dir, "analysis", "nn_sensitivity_mqfxfy", "clean_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path)
    val_col = "val_mse_scaled" if "val_mse_scaled" in df.columns else "val_mse"

    for activation in sorted(df["activation"].unique()):
        sub = df[df["activation"] == activation]
        layers = sorted(sub["layers"].unique())
        neurons = sorted(sub["neurons"].unique())
        pivot = (
            sub.pivot_table(
                index="layers", columns="neurons", values=val_col, aggfunc="mean"
            )
            .reindex(index=layers, columns=neurons)
            .to_numpy()
        )
        title = f"Validation MSE vs Layers/Neurons ({activation})"
        out_path = os.path.join(out_dir, f"heatmap_clean_{activation}.png")
        plot_heatmap(pivot, layers, neurons, title, out_path)

    print("Saved clean heatmaps to:", out_dir)


if __name__ == "__main__":
    main()
