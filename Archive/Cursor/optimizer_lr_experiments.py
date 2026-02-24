"""
Optimizer and learning-rate experiments for your best architecture.

Architecture fixed to: [256, 128]
Inputs : fx, fy, mz, q
Outputs: x1, x2, x3, x4, x5, x6, x7

We compare:
- Adam, fixed lr = 1e-3
- Adam + ReduceLROnPlateau
- AdamW, fixed lr = 1e-3
- AdamW + ReduceLROnPlateau
- AdamW + Cosine decay
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


np.random.seed(42)
tf.random.set_seed(42)


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

    return (
        X_train_s,
        X_test_s,
        y_train_s,
        y_test_s,
        scaler_y,
    )


def build_model(input_dim: int, output_dim: int):
    """Simple [256, 128] MLP without BN/Dropout (your best architecture)."""
    inputs = keras.Input(shape=(input_dim,), name="inputs")
    x = layers.Dense(256, activation="relu", name="dense1")(inputs)
    x = layers.Dense(128, activation="relu", name="dense2")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP_256_128")
    return model


def train_with_config(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler_y,
    optimizer_name: str,
    schedule_name: str,
    base_lr: float = 1e-3,
    epochs: int = 150,
    batch_size: int = 32,
):
    """Train one model with a specific optimizer + LR schedule."""
    print("\n" + "=" * 60)
    print(f"Optimizer: {optimizer_name}, LR schedule: {schedule_name}")
    print("=" * 60)

    model = build_model(X_train.shape[1], y_train.shape[1])

    # Optimizer
    if optimizer_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=base_lr)
    elif optimizer_name == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # LR schedule / callbacks
    callbacks = []
    if schedule_name == "none":
        pass
    elif schedule_name == "plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=0,
            )
        )
    elif schedule_name == "cosine":
        # Cosine decay over total steps (rough estimate using epochs * steps_per_epoch)
        steps_per_epoch = max(1, len(X_train) // batch_size)
        decay_steps = epochs * steps_per_epoch
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr, decay_steps=decay_steps
        )
        if optimizer_name == "adam":
            opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            opt = keras.optimizers.AdamW(
                learning_rate=lr_schedule, weight_decay=1e-4
            )
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    # Always use early stopping to keep runs stable
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, verbose=0
        )
    )

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

    # Evaluate
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

    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, Mean R²: {r2_mean:.4f}")

    return {
        "optimizer": optimizer_name,
        "schedule": schedule_name,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2_mean": r2_mean,
    }


def main():
    X_train, X_test, y_train, y_test, scaler_y = load_data("DATA.xlsx")

    configs = [
        ("adam", "none"),
        ("adam", "plateau"),
        ("adamw", "none"),
        ("adamw", "plateau"),
        ("adamw", "cosine"),
    ]

    results = []
    for opt_name, sched_name in configs:
        res = train_with_config(
            X_train,
            X_test,
            y_train,
            y_test,
            scaler_y,
            optimizer_name=opt_name,
            schedule_name=sched_name,
            base_lr=1e-3,
            epochs=150,
            batch_size=32,
        )
        results.append(res)

    df = pd.DataFrame(results).sort_values("rmse")
    print("\n" + "=" * 80)
    print("OPTIMIZER + LR SCHEDULE COMPARISON (ARCHITECTURE [256, 128])")
    print("=" * 80)
    print(df.to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"optimizer_lr_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()

