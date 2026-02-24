import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR

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
    # 70/15/15 split to match the NN pipeline.
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
        X_train_s,
        X_val_s,
        X_test_s,
        y_train_s,
        y_val_s,
        y_test_s,
        x_scaler,
        y_scaler,
    )


def parse_args():
    parser = ArgumentParser(
        description="Train Random Forest and SVR (RBF) with shared output pipeline."
    )
    parser.add_argument(
        "--models",
        choices=["rf", "svr", "both"],
        default="both",
        help="Which model(s) to train.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--rf-n-estimators", type=int, default=400)
    parser.add_argument("--rf-max-depth", type=int, default=0, help="0 means None.")
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)

    parser.add_argument("--svr-c", type=float, default=10.0)
    parser.add_argument("--svr-gamma", type=str, default="scale")
    parser.add_argument("--svr-epsilon", type=float, default=0.01)
    parser.add_argument(
        "--scaler",
        choices=["minmax", "robust", "standard"],
        default="minmax",
        help="Feature/target scaler type. Default is minmax (non z-score).",
    )
    parser.add_argument(
        "-w",
        "--weights",
        "--output-weights",
        dest="output_weights",
        type=str,
        default="",
        help="Comma-separated per-output weights for SVR, example: 1,1,1,1,1,1,2",
    )
    return parser.parse_args()


def slugify_float(value):
    txt = f"{value:.6g}"
    return txt.replace("-", "m").replace(".", "p")


def parse_output_weights(raw_weights, output_dim):
    if raw_weights is None or raw_weights.strip() == "":
        return None
    parts = [p.strip() for p in raw_weights.split(",") if p.strip()]
    if len(parts) != output_dim:
        raise ValueError(
            f"-w/--weights expects {output_dim} values (got {len(parts)}): {raw_weights}"
        )
    weights = np.array([float(p) for p in parts], dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError("-w/--weights values must be > 0 for weighted SVR scaling.")
    return weights


def build_arg_tag(model_name, args, output_dim):
    if model_name == "rf":
        depth = "none" if args.rf_max_depth == 0 else str(args.rf_max_depth)
        return (
            f"rf_scaler_{args.scaler}_n{args.rf_n_estimators}_d{depth}"
            f"_ss{args.rf_min_samples_split}_sl{args.rf_min_samples_leaf}"
        )

    gamma_slug = args.svr_gamma.replace(".", "p")
    weights = parse_output_weights(args.output_weights, output_dim)
    if weights is None:
        w_tag = "wdefault"
    else:
        w_tag = "w" + "-".join(slugify_float(float(w)) for w in weights)
    return (
        f"svr_rbf_scaler_{args.scaler}_c{slugify_float(args.svr_c)}_g{gamma_slug}"
        f"_e{slugify_float(args.svr_epsilon)}_{w_tag}"
    )


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


def evaluate_split(y_true_scaled, y_pred_scaled, y_scaler, split_name):
    y_true_unscaled = y_scaler.inverse_transform(y_true_scaled)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled)
    return {
        "split": split_name,
        "mse_scaled": float(mean_squared_error(y_true_scaled, y_pred_scaled)),
        "mae_scaled": float(mean_absolute_error(y_true_scaled, y_pred_scaled)),
        "r2_scaled": float(r2_score(y_true_scaled, y_pred_scaled)),
        "mse_unscaled": float(mean_squared_error(y_true_unscaled, y_pred_unscaled)),
        "mae_unscaled": float(mean_absolute_error(y_true_unscaled, y_pred_unscaled)),
        "r2_unscaled": float(r2_score(y_true_unscaled, y_pred_unscaled)),
        "y_true_unscaled": y_true_unscaled,
        "y_pred_unscaled": y_pred_unscaled,
    }


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


def save_split_predictions(split_result, output_cols, out_path):
    y_true = split_result["y_true_unscaled"]
    y_pred = split_result["y_pred_unscaled"]
    frame = {}
    for idx, col in enumerate(output_cols):
        frame[f"{col}_true"] = y_true[:, idx]
        frame[f"{col}_pred"] = y_pred[:, idx]
        frame[f"{col}_abs_error"] = np.abs(y_pred[:, idx] - y_true[:, idx])
    pd.DataFrame(frame).to_csv(out_path, index=False)


def save_fit_loss_plot(train_loss, val_loss, test_loss, history_dir, file_suffix):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["train", "validation", "test"]
    vals = [train_loss, val_loss, test_loss]
    bars = ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Loss (scaled MSE)")
    ax.set_title("Fit Loss by Split")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.6f}", ha="center", va="bottom")
    fig.tight_layout()
    plot_path = os.path.join(history_dir, f"training_loss_plot{file_suffix}.png")
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def train_svr_models(X_train, y_train, args, output_weights):
    estimators = []
    y_weighted = y_train * output_weights[np.newaxis, :]
    for j in range(y_train.shape[1]):
        est = SVR(
            kernel="rbf",
            C=args.svr_c,
            gamma=args.svr_gamma,
            epsilon=args.svr_epsilon,
        )
        est.fit(X_train, y_weighted[:, j])
        estimators.append(est)
    return estimators


def predict_svr_models(estimators, X, output_weights):
    pred_weighted = np.column_stack([est.predict(X) for est in estimators])
    return pred_weighted / output_weights[np.newaxis, :]


def build_model(model_name, args, seed):
    if model_name == "rf":
        max_depth = None if args.rf_max_depth == 0 else args.rf_max_depth
        return RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=max_depth,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )

    return None


def run_single_model(model_name, args, data_bundle, output_cols, input_cols):
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        x_scaler,
        y_scaler,
        output_root,
    ) = data_bundle

    runs_dir = os.path.join(output_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    arg_tag = build_arg_tag(model_name, args, len(output_cols))
    run_dir, run_name = build_unique_run_dir(runs_dir, arg_tag)
    file_suffix = f"_{arg_tag}"

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
        post_plots_dir,
        post_metrics_dir,
    ]:
        os.makedirs(folder, exist_ok=True)

    output_weights = parse_output_weights(args.output_weights, len(output_cols))

    if model_name == "svr":
        if output_weights is None:
            output_weights = np.ones(len(output_cols), dtype=np.float64)
        t0 = time.time()
        estimators = train_svr_models(X_train, y_train, args, output_weights)
        fit_time_sec = time.time() - t0
        pred_train = predict_svr_models(estimators, X_train, output_weights)
        pred_val = predict_svr_models(estimators, X_val, output_weights)
        pred_test = predict_svr_models(estimators, X_test, output_weights)
        model = {
            "model_type": "svr_rbf_weighted",
            "estimators": estimators,
            "output_weights": output_weights,
        }
    else:
        model = build_model(model_name, args, seed=args.seed)
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time_sec = time.time() - t0
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

    train_result = evaluate_split(y_train, pred_train, y_scaler, "train")
    val_result = evaluate_split(y_val, pred_val, y_scaler, "validation")
    test_result = evaluate_split(y_test, pred_test, y_scaler, "test")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"best_model{file_suffix}.joblib")
    model_versioned_path = os.path.join(models_dir, f"best_model{file_suffix}_{timestamp}.joblib")
    joblib.dump(model, model_path)
    joblib.dump(model, model_versioned_path)

    x_scaler_path = os.path.join(scalers_dir, f"x_scaler{file_suffix}.joblib")
    y_scaler_path = os.path.join(scalers_dir, f"y_scaler{file_suffix}.joblib")
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    scaler_stats = {
        "model": model_name,
        "scaler": args.scaler,
        "input_columns": input_cols,
        "output_columns": output_cols,
    }
    scaler_stats.update(scaler_stats_dict("x_scaler", x_scaler))
    scaler_stats.update(scaler_stats_dict("y_scaler", y_scaler))
    if model_name == "svr":
        scaler_stats["output_weights"] = output_weights.tolist()
    scaler_stats_path = os.path.join(scalers_dir, f"scaler_stats{file_suffix}.json")
    with open(scaler_stats_path, "w", encoding="utf-8") as f:
        json.dump(scaler_stats, f, indent=2)

    history_df = pd.DataFrame(
        [
            {
                "step": 1,
                "model": model_name,
                "fit_time_sec": fit_time_sec,
                "train_loss_scaled": train_result["mse_scaled"],
                "val_loss_scaled": val_result["mse_scaled"],
                "test_loss_scaled": test_result["mse_scaled"],
            }
        ]
    )
    history_path = os.path.join(history_dir, f"training_history{file_suffix}.csv")
    history_df.to_csv(history_path, index=False)
    loss_plot_path = save_fit_loss_plot(
        train_result["mse_scaled"],
        val_result["mse_scaled"],
        test_result["mse_scaled"],
        history_dir,
        file_suffix,
    )

    train_pred_path = os.path.join(predictions_dir, f"train_predictions{file_suffix}.csv")
    val_pred_path = os.path.join(predictions_dir, f"validation_predictions{file_suffix}.csv")
    test_pred_path = os.path.join(predictions_dir, f"test_predictions{file_suffix}.csv")
    save_split_predictions(train_result, output_cols, train_pred_path)
    save_split_predictions(val_result, output_cols, val_pred_path)
    save_split_predictions(test_result, output_cols, test_pred_path)

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
        title = f"Identity Plots ({split_name}) | model={model_name} | tag={arg_tag}"
        plot_identity(y_true, y_pred, output_cols, title, split_plot_path)

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
                "model": model_name,
                "arg_tag": arg_tag,
                "summary": post_summary,
            },
            f,
            indent=2,
        )

    metrics_summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "model": model_name,
        "arg_tag": arg_tag,
        "scaler": args.scaler,
        "fit_time_sec": fit_time_sec,
        "output_weights": output_weights.tolist() if model_name == "svr" else None,
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
            "loss_plot_path": loss_plot_path,
            "train_predictions_path": train_pred_path,
            "validation_predictions_path": val_pred_path,
            "test_predictions_path": test_pred_path,
            "postprocess_summary_path": post_summary_path,
            "postprocess_dir": post_dir,
            "file_suffix": file_suffix,
        },
    }
    metrics_path = os.path.join(metrics_dir, f"results_summary{file_suffix}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nModel: {model_name}")
    print(f"Run: {run_name}")
    if model_name == "svr":
        print(f"Output weights: {output_weights.tolist()}")
    print(
        f"Test MSE (scaled): {test_result['mse_scaled']:.10f} | "
        f"Test MSE (unscaled): {test_result['mse_unscaled']:.10f} | "
        f"Test R2 (unscaled): {test_result['r2_unscaled']:.10f}"
    )
    print(f"Artifacts: {run_dir}")


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "dataset", "DATA.xlsx")
    output_root = os.path.join(project_root, "output")

    X, y, input_cols, output_cols = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler = split_and_scale(
        X, y, seed=args.seed, scaler_name=args.scaler
    )
    data_bundle = (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        x_scaler,
        y_scaler,
        output_root,
    )

    selected = ["rf", "svr"] if args.models == "both" else [args.models]
    for model_name in selected:
        run_single_model(model_name, args, data_bundle, output_cols, input_cols)


if __name__ == "__main__":
    main()
