"""
Reza-style deep residual network for curve reconstruction on your data.

Inputs : [fx, fy, mz, q] (4 features)
Outputs: [x1, x2, x3, x4, x5, x6, x7] (7 targets)

Architecture (inspired by Khoshbakht et al. 2024):
- Fully connected feed-forward ANN
- 25 hidden layers
- 40 neurons per hidden layer
- ELU activation
- Residual (skip) connections: x -> Dense -> ELU -> Dense, then add skip

Training:
- Loss: Charbonnier loss (robust L2)
- Optimizer: AdamW + weight decay
- Input and output normalization
- Early stopping + learning rate reduction on plateau
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
    """Load and normalize data: 4 inputs, 7 outputs."""
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


def charbonnier_loss(eps: float = 1e-3):
    """Charbonnier loss: mean( sqrt((y - y_hat)^2 + eps^2) )."""

    def loss(y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.sqrt(tf.square(err) + eps**2))

    return loss


def build_reza_style_model(input_dim: int, output_dim: int):
    """Deep residual MLP: 25 hidden layers, 40 units, ELU, residual every 2 layers."""
    inputs = keras.Input(shape=(input_dim,), name="inputs")

    # First projection to 40 dims
    x = layers.Dense(40, activation="elu", name="dense_proj")(inputs)

    # We will create 12 residual blocks (2 layers each = 24 layers) + 1 extra layer = 25 total
    # Each residual block: x -> Dense(40) -> ELU -> Dense(40) ; add skip
    for b in range(12):
        shortcut = x
        x = layers.Dense(40, activation="elu", name=f"block{b+1}_dense1")(x)
        x = layers.Dense(40, activation=None, name=f"block{b+1}_dense2")(x)
        x = layers.Add(name=f"block{b+1}_add")([shortcut, x])
        x = layers.ELU(name=f"block{b+1}_elu")(x)

    # Extra hidden layer to reach ~25 layers total
    x = layers.Dense(40, activation="elu", name="dense_extra")(x)

    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RezaStyleDeepResNet")
    return model


def train_reza_style(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler_y,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 250,
    batch_size: int = 32,
):
    """Train the Reza-style model and report metrics."""
    model = build_reza_style_model(X_train.shape[1], y_train.shape[1])
    model.summary(print_fn=lambda x: None)  # suppress long summary in output

    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)

    model.compile(optimizer=opt, loss=charbonnier_loss(eps=1e-3), metrics=["mae"])

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True, verbose=0
        ),
    ]

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

    print("\n" + "=" * 80)
    print("REZA-STYLE DEEP RESIDUAL NETWORK RESULTS (ON YOUR DATA)")
    print("=" * 80)
    print(f"MSE : {mse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Mean R²: {r2_mean:.4f}")
    print(f"Per-output R²: {[f'{r:.4f}' for r in r2_scores]}")

    # Save summary CSV entry
    row = {
        "Architecture": "RezaStyle_25x40_Res_ELU_Charbonnier",
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "Mean_R2": r2_mean,
    }
    df = pd.DataFrame([row])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"reza_style_results_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return row


def main():
    X_train, X_test, y_train, y_test, scaler_y = load_data("DATA.xlsx")
    # For reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    train_reza_style(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_y,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=300,
        batch_size=32,
    )


if __name__ == "__main__":
    main()

