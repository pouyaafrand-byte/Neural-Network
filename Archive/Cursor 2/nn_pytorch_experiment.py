import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, layers=4, width=64, dropout=0.1):
        super().__init__()
        blocks = []
        dim = in_dim
        for _ in range(layers):
            blocks.append(nn.Linear(dim, width))
            blocks.append(nn.BatchNorm1d(width))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            dim = width
        blocks.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def load_data(path):
    df = pd.read_excel(path)
    input_cols = ["mz", "q", "fx", "fy"]
    output_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    base_X = df[input_cols].to_numpy()
    fx = df["fx"].to_numpy()
    fy = df["fy"].to_numpy()
    mz = df["mz"].to_numpy()
    q = df["q"].to_numpy()

    force_mag = np.sqrt(fx ** 2 + fy ** 2)
    force_angle = np.arctan2(fy, fx)
    mz_over_q = mz / (np.abs(q) + 1e-6)
    fx_fy = fx * fy
    fx_mz = fx * mz
    fy_mz = fy * mz

    X = np.column_stack(
        [
            base_X,
            force_mag,
            force_angle,
            mz_over_q,
            fx_fy,
            fx_mz,
            fy_mz,
        ]
    )
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
        Y_test,
    )


def train_model(model, train_loader, val_loader, epochs=300, lr=1e-3, wd=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None
    patience = 25
    patience_left = patience

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def evaluate(model, X_test, Y_test_scaled, Y_test_raw, y_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
    mse_scaled = mean_squared_error(Y_test_scaled, preds)
    preds_unscaled = y_scaler.inverse_transform(preds)
    mse_unscaled = mean_squared_error(Y_test_raw, preds_unscaled)
    return mse_scaled, mse_unscaled, preds_unscaled


def run_multi_output_configs(
    X_train, X_val, X_test, Y_train, Y_val, Y_test_scaled, Y_test_raw, y_scaler
):
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    configs = [
        {"layers": 8, "width": 128, "dropout": 0.2},
        {"layers": 10, "width": 128, "dropout": 0.2},
        {"layers": 8, "width": 192, "dropout": 0.2},
        {"layers": 10, "width": 192, "dropout": 0.2},
        {"layers": 12, "width": 256, "dropout": 0.2},
    ]

    best = {"val": np.inf}
    results = []

    for cfg in configs:
        model = MLP(
            in_dim=X_train.shape[1],
            out_dim=Y_train.shape[1],
            layers=cfg["layers"],
            width=cfg["width"],
            dropout=cfg["dropout"],
        )
        model, best_val = train_model(
            model, train_loader, val_loader, epochs=500, lr=2e-3, wd=1e-4
        )
        mse_scaled, mse_unscaled, _ = evaluate(
            model, X_test, Y_test_scaled, Y_test_raw, y_scaler
        )
        results.append(
            {
                "layers": cfg["layers"],
                "width": cfg["width"],
                "dropout": cfg["dropout"],
                "val_mse_scaled": best_val,
                "test_mse_scaled": mse_scaled,
                "test_mse_unscaled": mse_unscaled,
            }
        )
        if best_val < best["val"]:
            best = {
                "val": best_val,
                "test_scaled": mse_scaled,
                "test_unscaled": mse_unscaled,
                "cfg": cfg,
            }

    return best, results


def run_single_output_models(X_train, X_val, X_test, Y_train, Y_val, Y_test_scaled, Y_test_raw, y_scaler):
    preds = []
    best_vals = []
    for idx in range(Y_train.shape[1]):
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(Y_train[:, [idx]]).float()
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(Y_val[:, [idx]]).float()
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        model = MLP(in_dim=X_train.shape[1], out_dim=1, layers=5, width=96, dropout=0.2)
        model, best_val = train_model(model, train_loader, val_loader, epochs=400, lr=2e-3, wd=1e-4)
        best_vals.append(best_val)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
        preds.append(pred)

    preds = np.hstack(preds)
    mse_scaled = mean_squared_error(Y_test_scaled, preds)
    preds_unscaled = y_scaler.inverse_transform(preds)
    mse_unscaled = mean_squared_error(Y_test_raw, preds_unscaled)
    return float(np.mean(best_vals)), mse_scaled, mse_unscaled, preds_unscaled


def main():
    set_seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "DATA.xlsx")
    out_dir = os.path.join(base_dir, "analysis", "nn_pytorch_experiment")
    os.makedirs(out_dir, exist_ok=True)

    X, Y, input_cols, output_cols = load_data(data_path)
    (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test_scaled,
        _x_scaler,
        y_scaler,
        Y_test_raw,
    ) = split_and_scale(X, Y)

    multi_best, multi_results = run_multi_output_configs(
        X_train, X_val, X_test, Y_train, Y_val, Y_test_scaled, Y_test_raw, y_scaler
    )
    single_val, single_test_scaled, single_test_unscaled, _ = run_single_output_models(
        X_train, X_val, X_test, Y_train, Y_val, Y_test_scaled, Y_test_raw, y_scaler
    )

    report = [
        f"Multi-output best config: {multi_best['cfg']}",
        f"Multi-output best val MSE (scaled): {multi_best['val']}",
        f"Multi-output test MSE (scaled): {multi_best['test_scaled']}",
        f"Multi-output test MSE (unscaled): {multi_best['test_unscaled']}",
        "",
        f"Single-output avg best val MSE (scaled): {single_val}",
        f"Single-output test MSE (scaled): {single_test_scaled}",
        f"Single-output test MSE (unscaled): {single_test_unscaled}",
    ]

    results_path = os.path.join(out_dir, "multi_output_grid.csv")
    pd.DataFrame(multi_results).to_csv(results_path, index=False)

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()
