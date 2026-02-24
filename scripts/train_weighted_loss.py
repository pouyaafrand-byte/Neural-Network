import json
import os
from argparse import ArgumentParser
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    preferred_inputs = ["fx", "fy", "fz", "mz", "q"]
    fallback_inputs = ["mz", "q", "fx", "fy"]
    output_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]

    available_preferred = [c for c in preferred_inputs if c in df.columns]
    available_fallback = [c for c in fallback_inputs if c in df.columns]
    available_outputs = [c for c in output_cols if c in df.columns]

    if len(available_outputs) != len(output_cols):
        missing = [c for c in output_cols if c not in df.columns]
        raise ValueError(f"Missing output columns in DATA.xlsx: {missing}")

    input_cols = available_preferred if len(available_preferred) >= 4 else available_fallback
    if len(input_cols) < 4:
        raise ValueError(
            "Could not find enough input columns. Expected at least 4 of: "
            f"{preferred_inputs}"
        )

    X = df[input_cols].to_numpy(dtype=float)
    y = df[output_cols].to_numpy(dtype=float)
    return X, y, input_cols, output_cols


def build_scaler(scaler_name):
    if scaler_name == "standard":
        return StandardScaler(), StandardScaler()
    if scaler_name == "robust":
        return RobustScaler(), RobustScaler()
    if scaler_name == "minmax":
        return MinMaxScaler(), MinMaxScaler()
    raise ValueError(f"Unsupported scaler: {scaler_name}")


def split_and_scale(X, y, seed=42, scaler_name="minmax"):
    # 70/15/15 split: first keep 70% train, then split remaining 30% equally.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )

    x_scaler, y_scaler = build_scaler(scaler_name)

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_s,
        X_val_s,
        X_test_s,
        y_train_s,
        y_val_s,
        y_test_s,
        x_scaler,
        y_scaler,
    )


class TorchMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=8, width=20):
        super().__init__()
        blocks = []
        dim = in_dim
        for _ in range(hidden_layers):
            blocks.append(nn.Linear(dim, width))
            blocks.append(nn.ReLU())
            dim = width
        blocks.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def weighted_charbonnier_loss(pred, target, output_weights, eps=1e-3):
    diff = pred - target
    per_elem = torch.sqrt(diff * diff + eps * eps)
    per_output = per_elem.mean(dim=0)
    return torch.sum(per_output * output_weights) / torch.sum(output_weights)


def weighted_mse_loss(pred, target, output_weights):
    diff = pred - target
    per_elem = diff * diff
    per_output = per_elem.mean(dim=0)
    return torch.sum(per_output * output_weights) / torch.sum(output_weights)


def parse_args():
    parser = ArgumentParser(description="Train best NN with selectable loss.")
    parser.add_argument(
        "--loss",
        choices=["mse", "charbonnier"],
        default="charbonnier",
        help="Loss function to optimize.",
    )
    parser.add_argument(
        "-mse",
        "--mse",
        action="store_true",
        help="Shortcut flag: use MSE loss.",
    )
    parser.add_argument(
        "-charbonnier",
        "--charbonnier",
        action="store_true",
        help="Shortcut flag: use Charbonnier loss.",
    )
    parser.add_argument(
        "--charbonnier-eps",
        type=float,
        default=1e-3,
        help="Epsilon for Charbonnier loss.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        "--output-weights",
        dest="output_weights",
        type=str,
        default="",
        help=(
            "Comma-separated output weights for weighted loss (MSE or Charbonnier), "
            "example: 1,1,1,1,1,1,2"
        ),
    )
    parser.add_argument(
        "--scaler",
        choices=["minmax", "robust", "standard"],
        default="minmax",
        help="Feature/target scaler type. Default is minmax (non z-score).",
    )
    return parser.parse_args()


def resolve_loss_config(args):
    if args.mse:
        loss_name = "mse"
    elif args.charbonnier:
        loss_name = "charbonnier"
    else:
        loss_name = args.loss

    if loss_name == "mse":
        return {
            "loss_name": "mse",
            "file_suffix": "_mse",
            "loss_fn": nn.MSELoss(),
            "eps": None,
        }

    eps = float(args.charbonnier_eps)
    return {
        "loss_name": "charbonnier",
        "file_suffix": "_charbonnier",
        "loss_fn": lambda pred, target: charbonnier_loss(pred, target, eps=eps),
        "eps": eps,
    }


def parse_output_weights(raw_weights, output_dim):
    if raw_weights is None or raw_weights.strip() == "":
        return None
    parts = [p.strip() for p in raw_weights.split(",") if p.strip()]
    if len(parts) != output_dim:
        raise ValueError(
            f"-w/--weights expects {output_dim} values (got {len(parts)}): {raw_weights}"
        )
    weights = np.array([float(p) for p in parts], dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError("-w/--weights must be non-negative.")
    if float(weights.sum()) <= 0:
        raise ValueError("-w/--weights must not all be zero.")
    return weights


def slugify_float(value):
    txt = f"{value:.6g}"
    txt = txt.replace("-", "m").replace(".", "p")
    return txt


def make_weights_slug(weights):
    if weights is None:
        return "wdefault"
    return "w" + "-".join(slugify_float(float(w)) for w in weights)


def build_arg_tag(loss_name, eps, weights, scaler_name):
    parts = [loss_name]
    parts.append(f"scaler_{scaler_name}")
    if eps is not None:
        parts.append(f"eps{slugify_float(eps)}")
    parts.append(make_weights_slug(weights))
    return "_".join(parts)


def build_unique_run_dir(base_runs_dir, arg_tag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{arg_tag}_{timestamp}"
    run_dir = os.path.join(base_runs_dir, run_name)
    idx = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(base_runs_dir, f"{run_name}_{idx}")
        idx += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, os.path.basename(run_dir)


def scaler_stats_dict(prefix, scaler):
    stats = {"type": type(scaler).__name__}
    if hasattr(scaler, "mean_"):
        stats["mean"] = scaler.mean_.tolist()
    if hasattr(scaler, "scale_"):
        stats["scale"] = scaler.scale_.tolist()
    if hasattr(scaler, "center_"):
        stats["center"] = scaler.center_.tolist()
    if hasattr(scaler, "data_min_"):
        stats["data_min"] = scaler.data_min_.tolist()
    if hasattr(scaler, "data_max_"):
        stats["data_max"] = scaler.data_max_.tolist()
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def build_sparkline(values, width=20):
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    tail = values[-width:]
    vmin = min(tail)
    vmax = max(tail)
    if abs(vmax - vmin) < 1e-12:
        return blocks[0] * len(tail)
    chars = []
    for v in tail:
        idx = int((v - vmin) / (vmax - vmin) * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


def render_progress(
    epoch,
    total_epochs,
    train_mse,
    val_mse,
    test_mse,
    best_val,
    patience_left,
    train_hist,
    test_hist,
):
    width = 34
    progress = epoch / total_epochs
    filled = int(width * progress)
    empty = width - filled
    bar = color_text("█" * filled, "96") + color_text("░" * empty, "90")
    pct = f"{progress * 100:6.2f}%"

    line = (
        f"\r{color_text('Epoch', '94')} {epoch:>3}/{total_epochs:<3} "
        f"|{bar}| {color_text(pct, '92')} "
        f"| train_loss={train_mse:>10.6f} "
        f"| val_loss={val_mse:>10.6f} "
        f"| test_loss={test_mse:>10.6f} "
        f"| best={best_val:>10.6f} "
        f"| patience={patience_left:>2} "
        f"| {color_text('tr', '96')}:{build_sparkline(train_hist)} "
        f"{color_text('te', '93')}:{build_sparkline(test_hist)}"
    )
    print(line, end="", flush=True)


def evaluate_split(model, X_scaled, y_scaled, y_scaler, split_name):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_scaled).float().to(device)).cpu().numpy()
    pred_unscaled = y_scaler.inverse_transform(pred_scaled)
    y_unscaled = y_scaler.inverse_transform(y_scaled)

    return {
        "split": split_name,
        "mse_scaled": float(mean_squared_error(y_scaled, pred_scaled)),
        "mae_scaled": float(mean_absolute_error(y_scaled, pred_scaled)),
        "r2_scaled": float(r2_score(y_scaled, pred_scaled)),
        "mse_unscaled": float(mean_squared_error(y_unscaled, pred_unscaled)),
        "mae_unscaled": float(mean_absolute_error(y_unscaled, pred_unscaled)),
        "r2_unscaled": float(r2_score(y_unscaled, pred_unscaled)),
        "y_true_unscaled": y_unscaled,
        "y_pred_unscaled": pred_unscaled,
    }


def save_split_predictions(split_result, output_cols, predictions_dir, file_suffix):
    y_true = split_result["y_true_unscaled"]
    y_pred = split_result["y_pred_unscaled"]
    frame = {}
    for idx, col in enumerate(output_cols):
        frame[f"{col}_true"] = y_true[:, idx]
        frame[f"{col}_pred"] = y_pred[:, idx]
        frame[f"{col}_abs_error"] = np.abs(y_pred[:, idx] - y_true[:, idx])

    out_path = os.path.join(
        predictions_dir, f"{split_result['split']}_predictions{file_suffix}.csv"
    )
    pd.DataFrame(frame).to_csv(out_path, index=False)
    return out_path


def save_loss_history_plot(history_df, history_dir, file_suffix, timestamp):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_df["epoch"], history_df["train_loss_scaled"], label="train", linewidth=2.0)
    ax.plot(history_df["epoch"], history_df["val_loss_scaled"], label="validation", linewidth=2.0)
    ax.plot(history_df["epoch"], history_df["test_loss_scaled"], label="test", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (scaled)")
    ax.set_title("Training/Validation/Test Loss History")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(history_dir, f"training_loss_plot{file_suffix}.png")
    plot_versioned_path = os.path.join(
        history_dir, f"training_loss_plot{file_suffix}_{timestamp}.png"
    )
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    fig.savefig(plot_versioned_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path, plot_versioned_path


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
    rows.append(
        {
            "output": "aggregate",
            "mse": float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))),
            "mae": float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),
            "r2": float(r2_score(y_true.reshape(-1), y_pred.reshape(-1))),
        }
    )
    return pd.DataFrame(rows)


def plot_identity(y_true, y_pred, output_cols, title, out_path):
    n_outputs = len(output_cols)
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(11, 11))
    axes = axes.ravel()

    for i, col in enumerate(output_cols):
        ax = axes[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]
        low = min(float(yt.min()), float(yp.min()))
        high = max(float(yt.max()), float(yp.max()))
        ax.scatter(yt, yp, s=10, alpha=0.70, color="#1f77b4", edgecolors="none")
        ax.plot([low, high], [low, high], "--", color="#d62728", linewidth=1.2)
        ax.set_title(f"{col} (R2={r2_score(yt, yp):.4f})", fontsize=10)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.grid(alpha=0.25)

    agg_ax = axes[n_outputs]
    yt_flat = y_true.reshape(-1)
    yp_flat = y_pred.reshape(-1)
    low = min(float(yt_flat.min()), float(yp_flat.min()))
    high = max(float(yt_flat.max()), float(yp_flat.max()))
    agg_ax.scatter(yt_flat, yp_flat, s=8, alpha=0.55, color="#2ca02c", edgecolors="none")
    agg_ax.plot([low, high], [low, high], "--", color="#d62728", linewidth=1.2)
    agg_ax.set_title(f"aggregate (R2={r2_score(yt_flat, yp_flat):.4f})", fontsize=10)
    agg_ax.set_xlabel("Ground Truth")
    agg_ax.set_ylabel("Prediction")
    agg_ax.grid(alpha=0.25)

    for j in range(n_outputs + 1, rows * cols):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    loss_cfg = resolve_loss_config(args)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "dataset", "DATA.xlsx")
    output_root = os.path.join(project_root, "output")
    runs_dir = os.path.join(output_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    loss_name = loss_cfg["loss_name"]
    file_suffix = loss_cfg["file_suffix"]
    eps = loss_cfg["eps"]

    X, y, input_cols, output_cols = load_data(data_path)
    (
        _X_train_raw,
        _X_val_raw,
        _X_test_raw,
        _y_train_raw,
        _y_val_raw,
        _y_test_raw,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        x_scaler,
        y_scaler,
    ) = split_and_scale(X, y, seed=42, scaler_name=args.scaler)

    output_weights_np = parse_output_weights(args.output_weights, len(output_cols))
    output_weights = None
    if output_weights_np is not None:
        output_weights = torch.from_numpy(output_weights_np)

    if loss_name == "mse":
        if output_weights is None:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = lambda pred, target: weighted_mse_loss(
                pred, target, output_weights=output_weights.to(pred.device)
            )
    else:
        if output_weights is None:
            loss_fn = lambda pred, target: charbonnier_loss(pred, target, eps=eps)
        else:
            loss_fn = lambda pred, target: weighted_charbonnier_loss(
                pred, target, output_weights=output_weights.to(pred.device), eps=eps
            )

    arg_tag = build_arg_tag(loss_name, eps, output_weights_np, args.scaler)
    file_suffix = f"_{arg_tag}"
    run_dir, run_name = build_unique_run_dir(runs_dir, arg_tag)
    models_dir = os.path.join(run_dir, "models")
    scalers_dir = os.path.join(run_dir, "scalers")
    metrics_dir = os.path.join(run_dir, "metrics")
    history_dir = os.path.join(run_dir, "history")
    predictions_dir = os.path.join(run_dir, "predictions")
    post_dir = os.path.join(run_dir, "postprocessing")
    post_plots_dir = os.path.join(post_dir, "identity_plots")
    post_metrics_dir = os.path.join(post_dir, "metrics")
    for folder in [
        models_dir,
        scalers_dir,
        metrics_dir,
        history_dir,
        predictions_dir,
        post_dir,
        post_plots_dir,
        post_metrics_dir,
    ]:
        os.makedirs(folder, exist_ok=True)

    max_epochs = 200
    patience = 25
    tolerance = 1e-8
    best_val = float("inf")
    best_epoch = 0
    patience_left = patience
    best_state = None
    history = []
    model = TorchMLP(
        in_dim=X_train.shape[1],
        out_dim=y_train.shape[1],
        hidden_layers=8,
        width=20,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        ),
        batch_size=64,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    print(color_text("\nStarting training with best heatmap architecture", "95"))
    print(color_text("=" * 90, "90"))
    print(
        "Architecture: hidden_layers=8, width=20, activation=relu, optimizer=AdamW"
    )
    if eps is None:
        print(f"Loss: {loss_name}")
    else:
        print(f"Loss: {loss_name} (eps={eps})")
    if output_weights_np is not None:
        print(f"Output weights: {output_weights_np.tolist()}")
    print(
        f"Dataset: samples={len(X)} | train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
    )
    print(f"Scaler: {args.scaler}")
    print(f"Inputs: {input_cols}")
    print(f"Outputs: {output_cols}\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t).item())
            test_loss = float(loss_fn(model(X_test_t), y_test_t).item())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")

        history.append(
            {
                "epoch": epoch,
                "loss_name": loss_name,
                "train_loss_scaled": train_loss,
                "val_loss_scaled": val_loss,
                "test_loss_scaled": test_loss,
            }
        )

        if val_loss < best_val - tolerance:
            best_val = val_loss
            best_epoch = epoch
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1

        train_hist = [h["train_loss_scaled"] for h in history]
        test_hist = [h["test_loss_scaled"] for h in history]
        render_progress(
            epoch,
            max_epochs,
            train_loss,
            val_loss,
            test_loss,
            best_val,
            patience_left,
            train_hist,
            test_hist,
        )

        if patience_left <= 0:
            break
    print()

    if best_state is None:
        raise RuntimeError("Training finished without a valid best model checkpoint.")

    best_model = TorchMLP(
        in_dim=X_train.shape[1],
        out_dim=y_train.shape[1],
        hidden_layers=8,
        width=20,
    ).to(device)
    best_model.load_state_dict(best_state)

    train_result = evaluate_split(best_model, X_train, y_train, y_scaler, "train")
    val_result = evaluate_split(best_model, X_val, y_val, y_scaler, "validation")
    test_result = evaluate_split(best_model, X_test, y_test, y_scaler, "test")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"best_model{file_suffix}.pt")
    model_versioned_path = os.path.join(models_dir, f"best_model{file_suffix}_{timestamp}.pt")
    model_payload = {
        "model_class": "TorchMLP",
        "loss": loss_name,
        "loss_eps": eps,
        "in_dim": X_train.shape[1],
        "out_dim": y_train.shape[1],
        "hidden_layers": 8,
        "width": 20,
        "state_dict": best_model.state_dict(),
        "input_columns": input_cols,
        "output_columns": output_cols,
    }
    if eps is not None:
        model_payload["loss_eps"] = eps
    torch.save(model_payload, model_path)
    torch.save(model_payload, model_versioned_path)

    x_scaler_path = os.path.join(scalers_dir, f"x_scaler{file_suffix}.joblib")
    y_scaler_path = os.path.join(scalers_dir, f"y_scaler{file_suffix}.joblib")
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    scaler_stats = {"input_columns": input_cols, "output_columns": output_cols}
    scaler_stats.update(scaler_stats_dict("x_scaler", x_scaler))
    scaler_stats.update(scaler_stats_dict("y_scaler", y_scaler))
    scaler_stats["loss"] = loss_name
    scaler_stats["scaler"] = args.scaler
    if eps is not None:
        scaler_stats["loss_eps"] = eps
    if output_weights_np is not None:
        scaler_stats["output_weights"] = output_weights_np.tolist()
    scaler_stats_path = os.path.join(scalers_dir, f"scaler_stats{file_suffix}.json")
    with open(scaler_stats_path, "w", encoding="utf-8") as f:
        json.dump(scaler_stats, f, indent=2)

    history_df = pd.DataFrame(history)
    history_path = os.path.join(history_dir, f"training_history{file_suffix}.csv")
    history_versioned_path = os.path.join(
        history_dir, f"training_history{file_suffix}_{timestamp}.csv"
    )
    history_df.to_csv(history_path, index=False)
    history_df.to_csv(history_versioned_path, index=False)
    loss_plot_path, loss_plot_versioned_path = save_loss_history_plot(
        history_df, history_dir, file_suffix, timestamp
    )

    train_pred_path = save_split_predictions(train_result, output_cols, predictions_dir, file_suffix)
    val_pred_path = save_split_predictions(val_result, output_cols, predictions_dir, file_suffix)
    test_pred_path = save_split_predictions(test_result, output_cols, predictions_dir, file_suffix)

    weights_label = (
        ",".join(f"{float(w):.4g}" for w in output_weights_np)
        if output_weights_np is not None
        else "default"
    )
    post_title_meta = (
        f"loss={loss_name}"
        + (f", eps={eps:.4g}" if eps is not None else "")
        + f", weights={weights_label}"
    )
    post_summary = {}
    for split_name, split_result in [
        ("train", train_result),
        ("validation", val_result),
        ("test", test_result),
    ]:
        y_true = split_result["y_true_unscaled"]
        y_pred = split_result["y_pred_unscaled"]
        split_metrics_df = evaluate_predictions(y_true, y_pred, output_cols)
        split_metrics_path = os.path.join(
            post_metrics_dir, f"{split_name}_metrics{file_suffix}.csv"
        )
        split_metrics_df.to_csv(split_metrics_path, index=False)

        split_plot_path = os.path.join(
            post_plots_dir, f"{split_name}_identity_plot{file_suffix}.png"
        )
        plot_identity(
            y_true,
            y_pred,
            output_cols,
            f"Identity Plots ({split_name}) | {post_title_meta}",
            split_plot_path,
        )
        agg_row = split_metrics_df[split_metrics_df["output"] == "aggregate"].iloc[0]
        post_summary[split_name] = {
            "aggregate_mse": float(agg_row["mse"]),
            "aggregate_mae": float(agg_row["mae"]),
            "aggregate_r2": float(agg_row["r2"]),
            "metrics_csv": split_metrics_path,
            "identity_plot": split_plot_path,
        }
    post_summary_path = os.path.join(post_dir, f"postprocess_summary{file_suffix}.json")
    with open(post_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "loss_name": loss_name,
                "eps": eps,
                "output_weights": output_weights_np.tolist()
                if output_weights_np is not None
                else None,
                "summary": post_summary,
            },
            f,
            indent=2,
        )

    metrics_summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "architecture_source": "Archive/Cursor 2/analysis/nn_sensitivity/sensitivity_results.csv",
        "best_architecture": {
            "hidden_layers": 8,
            "neurons_per_layer": 20,
            "activation": "relu",
            "alpha_l2": 1e-5,
            "optimizer": "AdamW",
            "loss": loss_name,
            "scaler": args.scaler,
        },
        "training": {
            "max_epochs": max_epochs,
            "epochs_completed": int(history_df["epoch"].max()),
            "early_stopped": int(history_df["epoch"].max()) < max_epochs,
            "best_epoch": best_epoch,
            "best_val_loss_scaled": best_val,
            "patience": patience,
        },
        "results": {
            "train": {
                "mse_scaled": train_result["mse_scaled"],
                "mae_scaled": train_result["mae_scaled"],
                "r2_scaled": train_result["r2_scaled"],
                "mse_unscaled": train_result["mse_unscaled"],
                "mae_unscaled": train_result["mae_unscaled"],
                "r2_unscaled": train_result["r2_unscaled"],
            },
            "validation": {
                "mse_scaled": val_result["mse_scaled"],
                "mae_scaled": val_result["mae_scaled"],
                "r2_scaled": val_result["r2_scaled"],
                "mse_unscaled": val_result["mse_unscaled"],
                "mae_unscaled": val_result["mae_unscaled"],
                "r2_unscaled": val_result["r2_unscaled"],
            },
            "test": {
                "mse_scaled": test_result["mse_scaled"],
                "mae_scaled": test_result["mae_scaled"],
                "r2_scaled": test_result["r2_scaled"],
                "mse_unscaled": test_result["mse_unscaled"],
                "mae_unscaled": test_result["mae_unscaled"],
                "r2_unscaled": test_result["r2_unscaled"],
            },
        },
        "artifacts": {
            "model_path": model_path,
            "model_versioned_path": model_versioned_path,
            "x_scaler_path": x_scaler_path,
            "y_scaler_path": y_scaler_path,
            "scaler_stats_path": scaler_stats_path,
            "history_path": history_path,
            "history_versioned_path": history_versioned_path,
            "loss_plot_path": loss_plot_path,
            "loss_plot_versioned_path": loss_plot_versioned_path,
            "train_predictions_path": train_pred_path,
            "validation_predictions_path": val_pred_path,
            "test_predictions_path": test_pred_path,
            "predictions_dir": predictions_dir,
            "postprocess_summary_path": post_summary_path,
            "postprocess_dir": post_dir,
            "file_suffix": file_suffix,
        },
    }
    if eps is not None:
        metrics_summary["best_architecture"]["loss_eps"] = eps
    if output_weights_np is not None:
        metrics_summary["best_architecture"]["output_weights"] = output_weights_np.tolist()

    metrics_path = os.path.join(metrics_dir, f"results_summary{file_suffix}.json")
    metrics_versioned_path = os.path.join(
        metrics_dir, f"results_summary{file_suffix}_{timestamp}.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(metrics_versioned_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    report_lines = [
        "Training completed successfully.",
        f"Run name: {run_name}",
        f"Loss: {loss_name}" + (f" (eps={eps})" if eps is not None else ""),
        f"Output weights: {weights_label}",
        f"Best epoch: {best_epoch}",
        f"Best validation loss (scaled): {best_val:.10f}",
        "",
        "Final metrics (best checkpoint):",
        f"Train  - MSE(scaled): {train_result['mse_scaled']:.10f} | MSE(unscaled): {train_result['mse_unscaled']:.10f} | R2(unscaled): {train_result['r2_unscaled']:.10f}",
        f"Val    - MSE(scaled): {val_result['mse_scaled']:.10f} | MSE(unscaled): {val_result['mse_unscaled']:.10f} | R2(unscaled): {val_result['r2_unscaled']:.10f}",
        f"Test   - MSE(scaled): {test_result['mse_scaled']:.10f} | MSE(unscaled): {test_result['mse_unscaled']:.10f} | R2(unscaled): {test_result['r2_unscaled']:.10f}",
        "",
        f"Loss history plot: {loss_plot_path}",
        f"Postprocess summary: {post_summary_path}",
        "",
        f"Artifacts saved under: {run_dir}",
    ]
    print(color_text("\n" + "\n".join(report_lines), "92"))


if __name__ == "__main__":
    main()
