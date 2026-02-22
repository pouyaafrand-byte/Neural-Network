"""
Test diamond-shaped neural network architectures and visualize curves.
Diamond shape: expanding then contracting (e.g., [20, 40, 20] or [30, 50, 30])
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DiamondArchitectureTester:
    def __init__(self, data_file='DATA.xlsx'):
        """Initialize the tester with data file."""
        self.data_file = data_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_original = None
        self.X_test_original = None
        self.y_train_original = None
        self.y_test_original = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.results = []
        
    def load_data(self):
        """Load and preprocess the Excel data."""
        print("Loading data from Excel file...")
        df = pd.read_excel(self.data_file)
        
        # Input features: fx, fy, mz, q
        input_cols = ['fx', 'fy', 'mz', 'q']
        # Output features: x1, x2, x3, x4, x5, x6, x7
        output_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
        
        # Check if columns exist (case-insensitive)
        df.columns = df.columns.str.strip().str.lower()
        
        # Find matching columns
        available_inputs = [col for col in input_cols if col in df.columns]
        available_outputs = [col for col in output_cols if col in df.columns]
        
        X = df[available_inputs].values
        y = df[available_outputs].values
        
        # Remove any rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[mask]
        y = y[mask]
        
        # Store original data for visualization
        self.X_train_original, self.X_test_original, self.y_train_original, self.y_test_original = \
            train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        self.X_train = self.scaler_X.fit_transform(X_train)
        self.X_test = self.scaler_X.transform(X_test)
        self.y_train = self.scaler_y.fit_transform(y_train)
        self.y_test = self.scaler_y.transform(y_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return True
    
    def create_model(self, architecture_name, layers_config, activation='relu', 
                     dropout_rate=0.0, l2_reg=0.0):
        """Create a neural network model with specified architecture."""
        model = keras.Sequential(name=architecture_name)
        
        # Input layer
        model.add(layers.Input(shape=(self.X_train.shape[1],)))
        
        # Hidden layers
        for i, neurons in enumerate(layers_config):
            model.add(layers.Dense(
                neurons,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                name=f'hidden_{i+1}'
            ))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(
            self.y_train.shape[1],
            activation='linear',
            name='output'
        ))
        
        return model
    
    def train_and_evaluate(self, model, epochs=150, batch_size=32, verbose=0):
        """Train and evaluate a model."""
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        # Evaluate on test set
        y_pred = model.predict(self.X_test, verbose=0)
        
        # Inverse transform predictions and actual values
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        y_test_original = self.scaler_y.inverse_transform(self.y_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        
        # R2 score for each output
        r2_scores = [r2_score(y_test_original[:, i], y_pred_original[:, i]) 
                     for i in range(y_test_original.shape[1])]
        r2_mean = np.mean(r2_scores)
        
        return {
            'model': model,
            'history': history,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_mean': r2_mean,
            'r2_scores': r2_scores,
            'y_pred': y_pred_original,
            'y_test': y_test_original
        }
    
    def plot_curve(self, x_values, label, color, linestyle='-', linewidth=2):
        """
        Plot the curve from the 7 output values.
        point1 = (0, 0)
        point2 = (x1, 0)
        point3 = (x2, x3)
        point4 = (x4, x5)
        point5 = (x6, x7)
        """
        x_coords = [0, x_values[0], x_values[1], x_values[3], x_values[5]]
        y_coords = [0, 0, x_values[2], x_values[4], x_values[6]]
        
        plt.plot(x_coords, y_coords, color=color, linestyle=linestyle, 
                linewidth=linewidth, label=label, marker='o', markersize=6)
        
        return x_coords, y_coords
    
    def test_architecture(self, architecture_name, layers_config, activation='relu', 
                         dropout_rate=0.0, l2_reg=0.0, epochs=150, batch_size=32, verbose=0):
        """Test a specific architecture."""
        print(f"\n{'='*60}")
        print(f"Testing: {architecture_name}")
        print(f"Architecture: {layers_config}")
        print(f"{'='*60}")
        
        model = self.create_model(architecture_name, layers_config, activation=activation,
                                 dropout_rate=dropout_rate, l2_reg=l2_reg)
        
        # Count parameters
        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        # Train and evaluate
        result = self.train_and_evaluate(model, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        result['architecture_name'] = architecture_name
        result['layers_config'] = layers_config
        result['total_params'] = total_params
        
        print(f"MSE: {result['mse']:.6f}")
        print(f"MAE: {result['mae']:.6f}")
        print(f"RMSE: {result['rmse']:.6f}")
        print(f"Mean R²: {result['r2_mean']:.4f}")
        
        # Plot training curves
        self.plot_training_curves(result, architecture_name)
        
        # Plot geometric curves for a few test samples
        self.plot_geometric_curves(result, architecture_name)
        
        self.results.append(result)
        
        return result
    
    def plot_training_curves(self, result, architecture_name):
        """Plot training and validation loss curves."""
        history = result['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title(f'Training Curves - {architecture_name}', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MAE curve
        axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title(f'MAE Curves - {architecture_name}', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        safe_name = architecture_name.replace(' ', '_').replace('/', '_')
        filename = f'training_curves_{safe_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {filename}")
        plt.close()
    
    def plot_geometric_curves(self, result, architecture_name, num_samples=5):
        """Plot geometric curves comparing predicted vs actual."""
        y_pred = result['y_pred']
        y_test = result['y_test']
        
        # Select random samples from test set
        np.random.seed(42)
        sample_indices = np.random.choice(len(y_test), min(num_samples, len(y_test)), replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]
        
        for idx, sample_idx in enumerate(sample_indices):
            ax = axes[idx]
            
            # Get actual and predicted values
            actual = y_test[sample_idx]
            predicted = y_pred[sample_idx]
            
            # Plot actual curve
            x_coords_actual = [0, actual[0], actual[1], actual[3], actual[5]]
            y_coords_actual = [0, 0, actual[2], actual[4], actual[6]]
            ax.plot(x_coords_actual, y_coords_actual, 'b-', linewidth=2.5, 
                   label='Actual Curve', marker='o', markersize=8, alpha=0.8)
            
            # Plot predicted curve
            x_coords_pred = [0, predicted[0], predicted[1], predicted[3], predicted[5]]
            y_coords_pred = [0, 0, predicted[2], predicted[4], predicted[6]]
            ax.plot(x_coords_pred, y_coords_pred, 'r--', linewidth=2, 
                   label='Predicted Curve', marker='s', markersize=6, alpha=0.8)
            
            # Mark points
            ax.scatter(x_coords_actual, y_coords_actual, c='blue', s=100, 
                      zorder=5, edgecolors='black', linewidths=1.5)
            ax.scatter(x_coords_pred, y_coords_pred, c='red', s=80, 
                      zorder=5, edgecolors='black', linewidths=1.5, marker='s')
            
            ax.set_xlabel('X Coordinate', fontsize=11)
            ax.set_ylabel('Y Coordinate', fontsize=11)
            ax.set_title(f'Sample {sample_idx+1}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.suptitle(f'Geometric Curves Comparison - {architecture_name}', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        safe_name = architecture_name.replace(' ', '_').replace('/', '_')
        filename = f'geometric_curves_{safe_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Geometric curves saved to: {filename}")
        plt.close()
    
    def run_comparison(self):
        """Run comparison of diamond-shaped architectures."""
        if self.X_train is None:
            print("Error: Data not loaded. Please run load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DIAMOND-SHAPED NEURAL NETWORK ARCHITECTURE COMPARISON")
        print("="*60)
        
        # Define diamond-shaped architectures (expanding then contracting)
        # Using neuron counts: 20, 30, 40, 50
        architectures = [
            # 2-layer diamond shapes
            ("Diamond_2L_20_40", [20, 40]),
            ("Diamond_2L_30_50", [30, 50]),
            ("Diamond_2L_40_60", [40, 60]),
            
            # 3-layer diamond shapes (symmetric)
            ("Diamond_3L_20_40_20", [20, 40, 20]),
            ("Diamond_3L_30_50_30", [30, 50, 30]),
            ("Diamond_3L_40_60_40", [40, 60, 40]),
            ("Diamond_3L_20_50_20", [20, 50, 20]),
            ("Diamond_3L_30_60_30", [30, 60, 30]),
            
            # 3-layer diamond shapes (asymmetric)
            ("Diamond_3L_20_40_30", [20, 40, 30]),
            ("Diamond_3L_30_50_40", [30, 50, 40]),
            ("Diamond_3L_40_50_30", [40, 50, 30]),
            
            # 4-layer diamond shapes (symmetric)
            ("Diamond_4L_20_40_40_20", [20, 40, 40, 20]),
            ("Diamond_4L_30_50_50_30", [30, 50, 50, 30]),
            ("Diamond_4L_40_60_60_40", [40, 60, 60, 40]),
            ("Diamond_4L_20_40_50_30", [20, 40, 50, 30]),
            ("Diamond_4L_30_50_60_40", [30, 50, 60, 40]),
            
            # 4-layer diamond shapes (expanding-contracting)
            ("Diamond_4L_20_30_40_30", [20, 30, 40, 30]),
            ("Diamond_4L_30_40_50_40", [30, 40, 50, 40]),
            ("Diamond_4L_40_50_60_50", [40, 50, 60, 50]),
        ]
        
        # Test each architecture
        for arch_name, layers_config in architectures:
            try:
                self.test_architecture(
                    arch_name,
                    layers_config,
                    epochs=150,
                    batch_size=32,
                    verbose=0
                )
            except Exception as e:
                print(f"Error testing {arch_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate comparison report
        self.generate_report()
    
    def generate_report(self):
        """Generate a comparison report."""
        if not self.results:
            print("No results to report.")
            return
        
        print("\n" + "="*80)
        print("COMPARISON REPORT - DIAMOND ARCHITECTURES")
        print("="*80)
        
        # Create summary dataframe
        summary_data = []
        for result in self.results:
            summary_data.append({
                'Architecture': result['architecture_name'],
                'Layers': str(result['layers_config']),
                'Parameters': result['total_params'],
                'MSE': result['mse'],
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'Mean R²': result['r2_mean']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('RMSE')
        
        print("\nResults sorted by RMSE (best to worst):")
        print(df_summary.to_string(index=False))
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'diamond_architectures_results_{timestamp}.csv'
        df_summary.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")
        
        # Find best model
        best_result = min(self.results, key=lambda x: x['rmse'])
        print(f"\n{'='*80}")
        print("BEST MODEL")
        print(f"{'='*80}")
        print(f"Architecture: {best_result['architecture_name']}")
        print(f"Layers: {best_result['layers_config']}")
        print(f"Parameters: {best_result['total_params']:,}")
        print(f"MSE: {best_result['mse']:.6f}")
        print(f"MAE: {best_result['mae']:.6f}")
        print(f"RMSE: {best_result['rmse']:.6f}")
        print(f"Mean R²: {best_result['r2_mean']:.4f}")
        
        # Create overall comparison plot
        self.plot_overall_comparison(df_summary)
        
        return df_summary
    
    def plot_overall_comparison(self, df_summary):
        """Create overall comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE comparison
        axes[0, 0].barh(range(len(df_summary)), df_summary['RMSE'].values, color='steelblue')
        axes[0, 0].set_yticks(range(len(df_summary)))
        axes[0, 0].set_yticklabels(df_summary['Architecture'], fontsize=8)
        axes[0, 0].set_xlabel('RMSE', fontsize=11)
        axes[0, 0].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # R² comparison
        axes[0, 1].barh(range(len(df_summary)), df_summary['Mean R²'].values, color='coral')
        axes[0, 1].set_yticks(range(len(df_summary)))
        axes[0, 1].set_yticklabels(df_summary['Architecture'], fontsize=8)
        axes[0, 1].set_xlabel('Mean R²', fontsize=11)
        axes[0, 1].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Parameters vs RMSE
        axes[1, 0].scatter(df_summary['Parameters'], df_summary['RMSE'], 
                          s=100, alpha=0.6, c='green', edgecolors='black', linewidths=1)
        axes[1, 0].set_xlabel('Number of Parameters', fontsize=11)
        axes[1, 0].set_ylabel('RMSE', fontsize=11)
        axes[1, 0].set_title('Parameters vs RMSE', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of layers vs RMSE
        num_layers = [len(eval(l)) for l in df_summary['Layers']]
        axes[1, 1].scatter(num_layers, df_summary['RMSE'], 
                          s=100, alpha=0.6, c='purple', edgecolors='black', linewidths=1)
        axes[1, 1].set_xlabel('Number of Hidden Layers', fontsize=11)
        axes[1, 1].set_ylabel('RMSE', fontsize=11)
        axes[1, 1].set_title('Number of Layers vs RMSE', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks([2, 3, 4])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f'diamond_comparison_plots_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {plot_filename}")
        plt.close()


def main():
    """Main function to run the comparison."""
    tester = DiamondArchitectureTester('DATA.xlsx')
    
    # Load data
    if not tester.load_data():
        print("Failed to load data. Please check the Excel file.")
        return
    
    # Run comparison
    tester.run_comparison()


if __name__ == "__main__":
    main()
