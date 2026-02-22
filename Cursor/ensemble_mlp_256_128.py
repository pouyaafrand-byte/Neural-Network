"""
Ensemble of MLP[256, 128] models for your dataset.

Idea:
- Train N models with the same architecture [256, 128]
  and optimizer (AdamW + ReduceLROnPlateau),
  but with different random seeds.
- Average their predictions on the test set.

Inputs : fx, fy, mz, q
Outputs: x1, x2, x3, x4, x5, x6, x7
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime


def load_data(path: str = "DATA.xlsx"):
    print("Loading data from Excel file...")
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()

    input_cols = ["fx", "fy", "mz", "q"]
    output_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]

    X = df[input_cols].values
    y = df[output_cols].values

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    print(f"Training set: {X_train_s.shape[0]} samples")
    print(f"Test set: {X_test_s.shape[0]} samples")

    return X_train_s, X_test_s, y_train_s, y_test_s, scaler_y


def build_model(input_dim: int, output_dim: int):
    """MLP with layers [256, 128]."""
    inputs = keras.Input(shape=(input_dim,), name="inputs")
    x = layers.Dense(256, activation="relu", name="dense1")(inputs)
    x = layers.Dense(128, activation="relu", name="dense2")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_256_128")
    return model


def train_single_model(
    seed: int,
    X_train,
    X_test,
    y_train,
    y_test,
    scaler_y,
    base_lr: float = 1e-3,
    epochs: int = 150,
    batch_size: int = 32,
):
    """Train one model with given random seed and return predictions + metrics."""
    print("\n" + "=" * 60)
    print(f"Training ensemble member with seed = {seed}")
    print("=" * 60)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = build_model(X_train.shape[1], y_train.shape[1])

    opt = keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=1e-4)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, verbose=0
        ),
    ]

    model.compile(optimizer=opt, loss="mse", metrics=["mae"])

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    y_test_orig = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2_scores = [
        r2_score(y_test_orig[:, i], y_pred_orig[:, i])
        for i in range(y_test_orig.shape[1])
    ]
    r2_mean = float(np.mean(r2_scores))

    print(f"Single model (seed {seed}) - MSE: {mse:.6f}, RMSE: {rmse:.6f}, Mean R²: {r2_mean:.4f}")

    return {
        "seed": seed,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2_mean": r2_mean,
        "r2_scores": r2_scores,
        "y_pred_orig": y_pred_orig,
        "y_test_orig": y_test_orig,
    }


def main():
    X_train, X_test, y_train, y_test, scaler_y = load_data("DATA.xlsx")

    # Number of models in ensemble
    num_models = 5
    seeds = [42, 43, 44, 45, 46]

    members = []
    for seed in seeds[:num_models]:
        res = train_single_model(
            seed,
            X_train,
            X_test,
            y_train,
            y_test,
            scaler_y,
            base_lr=1e-3,
            epochs=150,
            batch_size=32,
        )
        members.append(res)

    # Stack predictions and average
    all_preds = np.stack([m["y_pred_orig"] for m in members], axis=0)  # (M, N, 7)
    ensemble_pred = np.mean(all_preds, axis=0)  # (N, 7)
    y_test_orig = members[0]["y_test_orig"]

    mse_ens = mean_squared_error(y_test_orig, ensemble_pred)
    mae_ens = mean_absolute_error(y_test_orig, ensemble_pred)
    rmse_ens = np.sqrt(mse_ens)
    r2_scores_ens = [
        r2_score(y_test_orig[:, i], ensemble_pred[:, i])
        for i in range(y_test_orig.shape[1])
    ]
    r2_mean_ens = float(np.mean(r2_scores_ens))

    print("\n" + "=" * 80)
    print("ENSEMBLE RESULTS (5 x MLP[256,128], AdamW + Plateau)")
    print("=" * 80)
    print(f"Ensemble MSE : {mse_ens:.6f}")
    print(f"Ensemble MAE : {mae_ens:.6f}")
    print(f"Ensemble RMSE: {rmse_ens:.6f}")
    print(f"Ensemble Mean R²: {r2_mean_ens:.4f}")
    print(f"Ensemble per-output R²: {[f'{r:.4f}' for r in r2_scores_ens]}")

    # Save predictions and ground truth for plotting (correlation graphs)
    pred_df = pd.DataFrame(
        np.hstack([y_test_orig, ensemble_pred]),
        columns=[
            "y_true_x1",
            "y_true_x2",
            "y_true_x3",
            "y_true_x4",
            "y_true_x5",
            "y_true_x6",
            "y_true_x7",
            "y_pred_x1",
            "y_pred_x2",
            "y_pred_x3",
            "y_pred_x4",
            "y_pred_x5",
            "y_pred_x6",
            "y_pred_x7",
        ],
    )
    pred_path = "ensemble_256_128_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved ensemble predictions for plotting to: {pred_path}")

    # Also summarize individual models vs ensemble
    rows = []
    for m in members:
        rows.append(
            {
                "Type": f"single_seed_{m['seed']}",
                "MSE": m["mse"],
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "Mean_R2": m["r2_mean"],
            }
        )
    rows.append(
        {
            "Type": "ensemble_5",
            "MSE": mse_ens,
            "MAE": mae_ens,
            "RMSE": rmse_ens,
            "Mean_R2": r2_mean_ens,
        }
    )

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("SINGLE MODELS VS ENSEMBLE")
    print("=" * 80)
    print(df.to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"ensemble_256_128_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()

