"""
Neural Network Architecture Comparison Tool
Tests different architectures to find the best model for the given data.
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
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NeuralNetworkTester:
    def __init__(self, data_file='DATA.xlsx'):
        """Initialize the tester with data file."""
        self.data_file = data_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.results = []
        
    def load_data(self):
        """Load and preprocess the Excel data."""
        print("Loading data from Excel file...")
        df = pd.read_excel(self.data_file)
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Input features: fx, fy, fz, mz, q
        input_cols = ['fx', 'fy', 'fz', 'mz', 'q']
        # Output features: x1, x2, x3, x4, x5, x6, x7
        output_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
        
        # Check if columns exist (case-insensitive)
        df.columns = df.columns.str.strip().str.lower()
        
        # Find matching columns
        available_inputs = [col for col in input_cols if col in df.columns]
        available_outputs = [col for col in output_cols if col in df.columns]
        
        if not available_inputs:
            print("Warning: Input columns not found. Available columns:", df.columns.tolist())
            print("Please check column names in the Excel file.")
            return False
            
        if not available_outputs:
            print("Warning: Output columns not found. Available columns:", df.columns.tolist())
            print("Please check column names in the Excel file.")
            return False
        
        X = df[available_inputs].values
        y = df[available_outputs].values
        
        print(f"Input features: {available_inputs}")
        print(f"Output features: {available_outputs}")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        # Remove any rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X = X[mask]
        y = y[mask]
        
        print(f"Data after removing NaN: {X.shape[0]} samples")
        
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
        """
        Create a neural network model with specified architecture.
        
        Args:
            architecture_name: Name of the architecture
            layers_config: List of integers representing neurons in each hidden layer
            activation: Activation function
            dropout_rate: Dropout rate (0.0 means no dropout)
            l2_reg: L2 regularization factor
        """
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
    
    def train_and_evaluate(self, model, epochs=100, batch_size=32, verbose=0):
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
    
    def test_architecture(self, architecture_name, layers_config, activation='relu', 
                         dropout_rate=0.0, l2_reg=0.0, epochs=100, batch_size=32, verbose=0):
        """Test a specific architecture."""
        print(f"\n{'='*60}")
        print(f"Testing: {architecture_name}")
        print(f"Architecture: {layers_config}")
        print(f"{'='*60}")
        
        model = self.create_model(architecture_name, layers_config, activation=activation,
                                 dropout_rate=dropout_rate, l2_reg=l2_reg)
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        
        print(f"Total parameters: {total_params:,}")
        
        # Train and evaluate
        result = self.train_and_evaluate(model, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        result['architecture_name'] = architecture_name
        result['layers_config'] = layers_config
        result['total_params'] = total_params
        result['trainable_params'] = trainable_params
        
        print(f"MSE: {result['mse']:.6f}")
        print(f"MAE: {result['mae']:.6f}")
        print(f"RMSE: {result['rmse']:.6f}")
        print(f"Mean R²: {result['r2_mean']:.4f}")
        print(f"Individual R² scores: {[f'{r:.4f}' for r in result['r2_scores']]}")
        
        self.results.append(result)
        
        return result
    
    def run_comparison(self):
        """Run comparison of different architectures."""
        if self.X_train is None:
            print("Error: Data not loaded. Please run load_data() first.")
            return
        
        print("\n" + "="*60)
        print("NEURAL NETWORK ARCHITECTURE COMPARISON")
        print("="*60)
        
        # Define different architectures to test
        architectures = [
            # Simple architectures
            ("Single_Layer_32", [32]),
            ("Single_Layer_64", [64]),
            ("Single_Layer_128", [128]),
            
            # Two-layer architectures
            ("Two_Layer_32_16", [32, 16]),
            ("Two_Layer_64_32", [64, 32]),
            ("Two_Layer_128_64", [128, 64]),
            ("Two_Layer_256_128", [256, 128]),
            
            # Three-layer architectures
            ("Three_Layer_64_32_16", [64, 32, 16]),
            ("Three_Layer_128_64_32", [128, 64, 32]),
            ("Three_Layer_256_128_64", [256, 128, 64]),
            ("Three_Layer_512_256_128", [512, 256, 128]),
            
            # Four-layer architectures - More variations
            ("Four_Layer_64_32_16_8", [64, 32, 16, 8]),
            ("Four_Layer_128_64_32_16", [128, 64, 32, 16]),
            ("Four_Layer_256_128_64_32", [256, 128, 64, 32]),
            ("Four_Layer_512_256_128_64", [512, 256, 128, 64]),
            ("Four_Layer_128_96_64_32", [128, 96, 64, 32]),
            ("Four_Layer_256_192_128_64", [256, 192, 128, 64]),
            ("Four_Layer_512_384_256_128", [512, 384, 256, 128]),
            ("Four_Layer_1024_512_256_128", [1024, 512, 256, 128]),
            ("Four_Layer_256_128_96_64", [256, 128, 96, 64]),
            ("Four_Layer_512_256_192_128", [512, 256, 192, 128]),
            
            # Five-layer architectures - More variations
            ("Five_Layer_128_64_32_16_8", [128, 64, 32, 16, 8]),
            ("Five_Layer_256_128_64_32_16", [256, 128, 64, 32, 16]),
            ("Five_Layer_512_256_128_64_32", [512, 256, 128, 64, 32]),
            ("Five_Layer_256_192_128_64_32", [256, 192, 128, 64, 32]),
            ("Five_Layer_512_384_256_128_64", [512, 384, 256, 128, 64]),
            ("Five_Layer_1024_512_256_128_64", [1024, 512, 256, 128, 64]),
            ("Five_Layer_128_96_64_48_32", [128, 96, 64, 48, 32]),
            ("Five_Layer_256_192_128_96_64", [256, 192, 128, 96, 64]),
            ("Five_Layer_512_256_192_128_64", [512, 256, 192, 128, 64]),
            ("Five_Layer_256_128_96_64_32", [256, 128, 96, 64, 32]),
            
            # Wide architectures
            ("Wide_Two_Layer_512_256", [512, 256]),
            ("Wide_Three_Layer_512_256_128", [512, 256, 128]),
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
                continue
        
        # Generate comparison report
        self.generate_report()
    
    def generate_report(self):
        """Generate a comparison report."""
        if not self.results:
            print("No results to report.")
            return
        
        print("\n" + "="*80)
        print("COMPARISON REPORT")
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
        csv_filename = f'nn_comparison_results_{timestamp}.csv'
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
        
        # Plot comparison
        self.plot_comparison(df_summary)
        
        return df_summary
    
    def plot_comparison(self, df_summary):
        """Create visualization plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # RMSE comparison
            axes[0, 0].barh(range(len(df_summary)), df_summary['RMSE'].values)
            axes[0, 0].set_yticks(range(len(df_summary)))
            axes[0, 0].set_yticklabels(df_summary['Architecture'], fontsize=8)
            axes[0, 0].set_xlabel('RMSE')
            axes[0, 0].set_title('RMSE Comparison')
            axes[0, 0].invert_yaxis()
            
            # R² comparison
            axes[0, 1].barh(range(len(df_summary)), df_summary['Mean R²'].values)
            axes[0, 1].set_yticks(range(len(df_summary)))
            axes[0, 1].set_yticklabels(df_summary['Architecture'], fontsize=8)
            axes[0, 1].set_xlabel('Mean R²')
            axes[0, 1].set_title('R² Score Comparison')
            axes[0, 1].invert_yaxis()
            
            # Parameters vs RMSE
            axes[1, 0].scatter(df_summary['Parameters'], df_summary['RMSE'], s=100, alpha=0.6)
            axes[1, 0].set_xlabel('Number of Parameters')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].set_title('Parameters vs RMSE')
            axes[1, 0].set_xscale('log')
            
            # Number of layers vs RMSE
            num_layers = [len(eval(l)) for l in df_summary['Layers']]
            axes[1, 1].scatter(num_layers, df_summary['RMSE'], s=100, alpha=0.6)
            axes[1, 1].set_xlabel('Number of Hidden Layers')
            axes[1, 1].set_ylabel('RMSE')
            axes[1, 1].set_title('Number of Layers vs RMSE')
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f'nn_comparison_plots_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_filename}")
            plt.close()
            
        except Exception as e:
            print(f"Error creating plots: {str(e)}")


def main():
    """Main function to run the comparison."""
    tester = NeuralNetworkTester('DATA.xlsx')
    
    # Load data
    if not tester.load_data():
        print("Failed to load data. Please check the Excel file.")
        return
    
    # Run comparison
    tester.run_comparison()


if __name__ == "__main__":
    main()
