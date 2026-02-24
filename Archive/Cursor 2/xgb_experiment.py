import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def load_data(path):
    df = pd.read_excel(path)
    input_cols = ["mz", "q", "fx", "fy"]
    output_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    X = df[input_cols].to_numpy()
    Y = df[output_cols].to_numpy()
    return X, Y, input_cols, output_cols


def split_data(X, Y, seed=42):
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.30, random_state=seed
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.50, random_state=seed
    )
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def train_single_output(X_train, Y_train, X_val, Y_val):
    model = XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=0,
    )
    model.fit(X_train, Y_train)
    return model


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "DATA.xlsx")
    out_dir = os.path.join(base_dir, "analysis", "xgb_experiment")
    os.makedirs(out_dir, exist_ok=True)

    X, Y, input_cols, output_cols = load_data(data_path)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y)

    y_scaler = StandardScaler()
    Y_train_s = y_scaler.fit_transform(Y_train)
    Y_val_s = y_scaler.transform(Y_val)
    Y_test_s = y_scaler.transform(Y_test)

    preds_val = []
    preds_test = []
    for idx in range(Y_train_s.shape[1]):
        model = train_single_output(
            X_train, Y_train_s[:, idx], X_val, Y_val_s[:, idx]
        )
        preds_val.append(model.predict(X_val))
        preds_test.append(model.predict(X_test))

    pred_val_s = np.column_stack(preds_val)
    pred_test_s = np.column_stack(preds_test)

    val_mse_scaled = mean_squared_error(Y_val_s, pred_val_s)
    test_mse_scaled = mean_squared_error(Y_test_s, pred_test_s)

    pred_test_unscaled = y_scaler.inverse_transform(pred_test_s)
    test_mse_unscaled = mean_squared_error(Y_test, pred_test_unscaled)

    report = [
        f"Inputs: {input_cols}",
        f"Outputs: {output_cols}",
        f"Val MSE (scaled): {val_mse_scaled}",
        f"Test MSE (scaled): {test_mse_scaled}",
        f"Test MSE (unscaled): {test_mse_unscaled}",
    ]

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()
