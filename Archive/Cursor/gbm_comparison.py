"""
Gradient Boosting (XGBoost) comparison against neural networks.

Inputs:  fx, fy, mz, q
Outputs: x1, x2, x3, x4, x5, x6, x7

Trains one XGBoost regressor per output and reports:
- MSE, MAE, RMSE, R² for each output
- Mean metrics across all outputs
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime


def load_data(path: str = "DATA.xlsx"):
    """Load data and create a fixed train/test split (same for all models)."""
    print("Loading data from Excel file...")
    df = pd.read_excel(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    input_cols = ["fx", "fy", "mz", "q"]
    output_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]

    X = df[input_cols].values
    y = df[output_cols].values

    # Remove NaNs
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, input_cols, output_cols


def train_xgboost_models(X_train, X_test, y_train, y_test, output_cols):
    """Train one XGBRegressor per output."""
    results = []
    preds = np.zeros_like(y_test)

    for i, name in enumerate(output_cols):
        print("\n" + "=" * 60)
        print(f"Training XGBoost model for output: {name}")
        print("=" * 60)

        model = XGBRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=42,
        )

        model.fit(X_train, y_train[:, i])

        y_pred = model.predict(X_test)
        preds[:, i] = y_pred

        mse = mean_squared_error(y_test[:, i], y_pred)
        mae = mean_absolute_error(y_test[:, i], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test[:, i], y_pred)

        print(f"{name} - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")

        results.append(
            {
                "Output": name,
                "MSE": mse,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "model": model,
            }
        )

    return results, preds


def summarize_results(results, y_test, preds):
    """Summarize per-output and overall performance, save CSV."""
    rows = []
    for r in results:
        rows.append(
            {
                "Output": r["Output"],
                "MSE": r["MSE"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "R2": r["R2"],
            }
        )

    df_res = pd.DataFrame(rows)

    # Overall metrics across all outputs combined
    mse_all = mean_squared_error(y_test, preds)
    mae_all = mean_absolute_error(y_test, preds)
    rmse_all = np.sqrt(mse_all)
    r2_all_per_output = []
    for i in range(y_test.shape[1]):
        r2_all_per_output.append(r2_score(y_test[:, i], preds[:, i]))
    r2_mean = float(np.mean(r2_all_per_output))

    print("\n" + "=" * 80)
    print("XGBOOST RESULTS PER OUTPUT")
    print("=" * 80)
    print(df_res.to_string(index=False))

    print("\n" + "=" * 80)
    print("XGBOOST OVERALL METRICS (ALL 7 OUTPUTS TOGETHER)")
    print("=" * 80)
    print(f"Overall MSE : {mse_all:.6f}")
    print(f"Overall MAE : {mae_all:.6f}")
    print(f"Overall RMSE: {rmse_all:.6f}")
    print(f"Mean R²    : {r2_mean:.4f}")

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"xgboost_results_{timestamp}.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df_res, {
        "MSE": mse_all,
        "MAE": mae_all,
        "RMSE": rmse_all,
        "Mean_R2": r2_mean,
    }


def main():
    X_train, X_test, y_train, y_test, input_cols, output_cols = load_data("DATA.xlsx")
    results, preds = train_xgboost_models(X_train, X_test, y_train, y_test, output_cols)
    _, overall = summarize_results(results, y_test, preds)

    print("\n" + "=" * 80)
    print("SUMMARY VS NEURAL NETWORKS (for your thesis discussion)")
    print("=" * 80)
    print(
        "You can directly compare these XGBoost metrics with the best NN metrics\n"
        "from 'nn_comparison_results_*.csv' and 'diamond_architectures_results_*.csv'."
    )
    print(
        f"XGBoost overall RMSE: {overall['RMSE']:.6f}, "
        f"Mean R²: {overall['Mean_R2']:.4f}"
    )


if __name__ == "__main__":
    main()

