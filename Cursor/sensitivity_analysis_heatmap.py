"""
Sensitivity Analysis: Create heatmap of validation loss
Testing different architectures (layers × neurons) and activation functions
Similar to Reza's Figure 6(a)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SensitivityAnalysis:
    def __init__(self, data_file='DATA.xlsx'):
        """Initialize the sensitivity analysis."""
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
    
    def create_model(self, num_layers, neurons_per_layer, activation='relu', use_residual=False):
        """Create a neural network model with specified architecture."""
        model = keras.Sequential(name=f'NN_{num_layers}L_{neurons_per_layer}N')
        
        # Input layer
        model.add(layers.Input(shape=(self.X_train.shape[1],)))
        
        # Hidden layers
        for i in range(num_layers):
            if use_residual and i > 0:
                # For residual connections, we need to handle dimension matching
                # Simple approach: add residual only if dimensions match
                residual_input = model.layers[-1].output
                
                # Dense layer
                dense_out = layers.Dense(neurons_per_layer, activation=activation, 
                                        name=f'dense_{i+1}')(residual_input)
                
                # Add residual connection if dimensions match
                if i > 0 and residual_input.shape[-1] == neurons_per_layer:
                    dense_out = layers.Add(name=f'add_{i+1}')([residual_input, dense_out])
                
                # For Sequential, we'll use a simpler approach
                model.add(layers.Dense(neurons_per_layer, activation=activation, 
                                      name=f'hidden_{i+1}'))
            else:
                model.add(layers.Dense(neurons_per_layer, activation=activation, 
                                      name=f'hidden_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(self.y_train.shape[1], activation='linear', name='output'))
        
        return model
    
    def train_and_evaluate(self, model, epochs=100, batch_size=32, verbose=0):
        """Train and evaluate a model, return validation loss."""
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
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
        
        # Get best validation loss
        best_val_loss = min(history.history['val_loss'])
        
        return best_val_loss, history
    
    def run_sensitivity_analysis(self):
        """Run sensitivity analysis across architectures and activations."""
        if self.X_train is None:
            print("Error: Data not loaded. Please run load_data() first.")
            return
        
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS: VALIDATION LOSS HEATMAP")
        print("="*80)
        
        # Define parameter ranges (matching Reza's study)
        num_layers_list = [10, 15, 20, 25, 30]
        neurons_list = [30, 40, 50, 60, 70]
        activation_functions = {
            'ELU': 'elu',
            'LEAKY_RELU': 'leaky_relu',
            'RELU': 'relu',
            'TANH': 'tanh'
        }
        
        total_combinations = len(num_layers_list) * len(neurons_list) * len(activation_functions)
        print(f"\nTesting {total_combinations} combinations...")
        print("This may take a while...\n")
        
        count = 0
        for num_layers in num_layers_list:
            for neurons in neurons_list:
                for act_name, act_func in activation_functions.items():
                    count += 1
                    arch_name = f"{num_layers} layers, {neurons} neurons"
                    
                    print(f"[{count}/{total_combinations}] Testing: {arch_name} with {act_name}...", end=' ')
                    
                    try:
                        # Create model
                        model = self.create_model(num_layers, neurons, activation=act_func, use_residual=False)
                        
                        # Train and evaluate
                        val_loss, history = self.train_and_evaluate(model, epochs=100, verbose=0)
                        
                        self.results.append({
                            'Architecture': arch_name,
                            'Layers': num_layers,
                            'Neurons': neurons,
                            'Activation': act_name,
                            'Validation_Loss': val_loss
                        })
                        
                        print(f"Val Loss: {val_loss:.4f}")
                        
                        # Clear session to free memory
                        keras.backend.clear_session()
                        
                    except Exception as e:
                        print(f"ERROR: {str(e)}")
                        # Store a high loss value for failed cases
                        self.results.append({
                            'Architecture': arch_name,
                            'Layers': num_layers,
                            'Neurons': neurons,
                            'Activation': act_name,
                            'Validation_Loss': 100.0  # High penalty for failures
                        })
                        keras.backend.clear_session()
                        continue
        
        # Create heatmap
        self.create_heatmap()
        
        # Save results
        df_results = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'sensitivity_analysis_results_{timestamp}.csv'
        df_results.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")
        
        return df_results
    
    def create_heatmap(self):
        """Create heatmap visualization like Reza's Figure 6(a)."""
        if not self.results:
            print("No results to plot.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            values='Validation_Loss',
            index='Architecture',
            columns='Activation',
            aggfunc='mean'
        )
        
        # Sort architectures by layers then neurons
        df_temp = df[['Architecture', 'Layers', 'Neurons']].drop_duplicates()
        df_temp = df_temp.sort_values(['Layers', 'Neurons'])
        ordered_architectures = df_temp['Architecture'].values
        
        # Reorder pivot table rows
        pivot_data = pivot_data.reindex(ordered_architectures)
        
        # Create figure
        plt.figure(figsize=(10, 14))
        
        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red=high loss, green=low loss)
            cbar_kws={'label': 'Validation Loss'},
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=max(50, pivot_data.max().max())  # Cap at 50 for better visualization
        )
        
        plt.title('Validation Loss by Architecture and Activation Function', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Activation Function', fontsize=12, fontweight='bold')
        plt.ylabel('Architecture', fontsize=12, fontweight='bold')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'sensitivity_analysis_heatmap_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {filename}")
        plt.close()
        
        # Find best configuration
        best_result = df.loc[df['Validation_Loss'].idxmin()]
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"Architecture: {best_result['Architecture']}")
        print(f"Activation: {best_result['Activation']}")
        print(f"Validation Loss: {best_result['Validation_Loss']:.4f}")
        print(f"{'='*80}")


def main():
    """Main function to run sensitivity analysis."""
    analyzer = SensitivityAnalysis('DATA.xlsx')
    
    # Load data
    if not analyzer.load_data():
        print("Failed to load data. Please check the Excel file.")
        return
    
    # Run sensitivity analysis
    analyzer.run_sensitivity_analysis()


if __name__ == "__main__":
    main()
