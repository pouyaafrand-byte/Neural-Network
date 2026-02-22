"""
Test neural network architectural tricks on your tabular dataset:

- Batch Normalization (Dense -> BatchNorm -> ReLU)
- Dropout
- Residual connections (for deeper networks)

Inputs:  fx, fy, mz, q
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


np.random.seed(42)
tf.random.set_seed(42)


class AdvancedNNTester:
    def __init__(self, data_file: str = "DATA.xlsx"):
        self.data_file = data_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.results = []

    def load_data(self) -> bool:
        print("Loading data from Excel file...")
        df = pd.read_excel(self.data_file)

        input_cols = ["fx", "fy", "mz", "q"]
        output_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]

        df.columns = df.columns.str.strip().str.lower()

        X = df[input_cols].values
        y = df[output_cols].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[mask]
        y = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.X_train = self.scaler_X.fit_transform(X_train)
        self.X_test = self.scaler_X.transform(X_test)
        self.y_train = self.scaler_y.fit_transform(y_train)
        self.y_test = self.scaler_y.transform(y_test)

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        return True

    def _dense_block(
        self,
        x,
        units: int,
        use_batchnorm: bool,
        dropout_rate: float,
        name_prefix: str,
    ):
        """Dense -> (BatchNorm) -> ReLU -> (Dropout)."""
        x = layers.Dense(units, use_bias=not use_batchnorm, name=f"{name_prefix}_dense")(
            x
        )
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        x = layers.ReLU(name=f"{name_prefix}_relu")(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
        return x

    def create_model(
        self,
        architecture_name: str,
        layers_config,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
        use_residual: bool = False,
    ) -> keras.Model:
        """
        Create a model with optional BatchNorm, Dropout and Residual connections.
        Residual: add skip around two consecutive dense blocks if dimensions match.
        """
        inputs = keras.Input(shape=(self.X_train.shape[1],), name="inputs")
        x = inputs

        residual_buffer = None
        block_idx = 0

        for i, units in enumerate(layers_config):
            prev = x
            x = self._dense_block(
                x,
                units=units,
                use_batchnorm=use_batchnorm,
                dropout_rate=dropout_rate,
                name_prefix=f"block{i+1}",
            )

            if use_residual:
                # Simple residual every two blocks when dimensions match
                if residual_buffer is None:
                    residual_buffer = prev
                    block_idx = 1
                else:
                    # Only add residual if shapes match
                    if residual_buffer.shape[-1] == x.shape[-1]:
                        x = layers.Add(name=f"residual_{i+1}")([x, residual_buffer])
                    residual_buffer = None
                    block_idx = 0

        outputs = layers.Dense(
            self.y_train.shape[1], activation="linear", name="output"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=architecture_name)
        return model

    def train_and_evaluate(
        self, model: keras.Model, epochs: int = 150, batch_size: int = 32, verbose: int = 0
    ):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, verbose=0
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        )

        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose,
        )

        y_pred = model.predict(self.X_test, verbose=0)
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        y_test_original = self.scaler_y.inverse_transform(self.y_test)

        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2_scores = [
            r2_score(y_test_original[:, i], y_pred_original[:, i])
            for i in range(y_test_original.shape[1])
        ]
        r2_mean = float(np.mean(r2_scores))

        return {
            "model": model,
            "history": history,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2_mean": r2_mean,
            "r2_scores": r2_scores,
            "y_pred": y_pred_original,
            "y_test": y_test_original,
        }

    def test_architecture(
        self,
        architecture_name: str,
        layers_config,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
        use_residual: bool = False,
        epochs: int = 150,
        batch_size: int = 32,
        verbose: int = 0,
    ):
        print("\n" + "=" * 60)
        print(f"Testing: {architecture_name}")
        print(f"Layers: {layers_config}")
        print(
            f"BatchNorm: {use_batchnorm}, Dropout: {dropout_rate}, Residual: {use_residual}"
        )
        print("=" * 60)

        model = self.create_model(
            architecture_name,
            layers_config,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
        )

        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")

        result = self.train_and_evaluate(
            model, epochs=epochs, batch_size=batch_size, verbose=verbose
        )

        result["architecture_name"] = architecture_name
        result["layers_config"] = layers_config
        result["total_params"] = total_params

        print(f"MSE: {result['mse']:.6f}")
        print(f"MAE: {result['mae']:.6f}")
        print(f"RMSE: {result['rmse']:.6f}")
        print(f"Mean R²: {result['r2_mean']:.4f}")

        self.results.append(result)
        return result

    def run_comparison(self):
        if self.X_train is None:
            print("Error: data not loaded.")
            return

        print("\n" + "=" * 60)
        print("ADVANCED NN ARCHITECTURE TRICKS COMPARISON")
        print("=" * 60)

        # Focused set around the known good region (2–3 layers, 128–256 units)
        configs = [
            # Baseline-style (no tricks) around best NN
            ("Base_2L_256_128", [256, 128], False, 0.0, False),
            ("Base_3L_256_192_128", [256, 192, 128], False, 0.0, False),
            # BatchNorm only
            ("BN_2L_256_128", [256, 128], True, 0.0, False),
            ("BN_3L_256_192_128", [256, 192, 128], True, 0.0, False),
            # BatchNorm + small Dropout
            ("BN_Drop_2L_256_128", [256, 128], True, 0.1, False),
            ("BN_Drop_3L_256_192_128", [256, 192, 128], True, 0.1, False),
            # Deeper with residuals
            ("BN_Drop_Res_4L_256_256_128_128", [256, 256, 128, 128], True, 0.1, True),
            ("BN_Drop_Res_4L_256_192_192_128", [256, 192, 192, 128], True, 0.1, True),
        ]

        for name, layers_config, use_bn, drop, use_res in configs:
            try:
                self.test_architecture(
                    name,
                    layers_config,
                    use_batchnorm=use_bn,
                    dropout_rate=drop,
                    use_residual=use_res,
                    epochs=150,
                    batch_size=32,
                    verbose=0,
                )
            except Exception as e:
                print(f"Error testing {name}: {e}")
                continue

        self.generate_report()

    def generate_report(self):
        if not self.results:
            print("No results to report.")
            return

        rows = []
        for r in self.results:
            rows.append(
                {
                    "Architecture": r["architecture_name"],
                    "Layers": str(r["layers_config"]),
                    "Parameters": r["total_params"],
                    "MSE": r["mse"],
                    "MAE": r["mae"],
                    "RMSE": r["rmse"],
                    "Mean R²": r["r2_mean"],
                }
            )

        df = pd.DataFrame(rows).sort_values("RMSE")

        print("\n" + "=" * 80)
        print("ADVANCED NN TRICKS - COMPARISON REPORT")
        print("=" * 80)
        print(df.to_string(index=False))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"advanced_nn_tricks_results_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        best = df.iloc[0]
        print("\n" + "=" * 80)
        print("BEST ADVANCED NN MODEL")
        print("=" * 80)
        print(best)


def main():
    tester = AdvancedNNTester("DATA.xlsx")
    if not tester.load_data():
        return
    tester.run_comparison()


if __name__ == "__main__":
    main()

