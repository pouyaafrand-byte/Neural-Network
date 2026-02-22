# Neural Network Architecture Comparison Tool

This tool tests different neural network architectures to find the best model for your data.

## Inputs and Outputs

- **Inputs**: fx, fy, fz, mz, q (5 features)
- **Outputs**: x1, x2, x3, x4, x5, x6, x7 (7 targets)

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the comparison script:

```bash
python neural_network_comparison.py
```

## What it does

1. **Loads data** from `DATA.xlsx`
2. **Preprocesses data** (scaling, train/test split)
3. **Tests multiple architectures** with different:
   - Number of hidden layers (1 to 5 layers)
   - Number of neurons per layer (16 to 512)
   - Various configurations (narrow, wide, deep networks)
4. **Evaluates each model** using:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score (for each output and mean)
5. **Generates reports**:
   - CSV file with all results
   - Comparison plots
   - Best model identification

## Output Files

- `nn_comparison_results_YYYYMMDD_HHMMSS.csv`: Detailed results for all architectures
- `nn_comparison_plots_YYYYMMDD_HHMMSS.png`: Visualization plots

## Customization

You can modify the architectures tested by editing the `architectures` list in the `run_comparison()` method of the `NeuralNetworkTester` class.

## Features

- Automatic data scaling
- Early stopping to prevent overfitting
- Learning rate reduction
- Comprehensive evaluation metrics
- Visual comparison of architectures
- Parameter count tracking
