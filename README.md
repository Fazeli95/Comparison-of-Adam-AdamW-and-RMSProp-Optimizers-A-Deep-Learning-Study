# Comparison-of-Adam-AdamW-and-RMSProp-Optimizers-A-Deep-Learning-Study

This repository contains three Jupyter notebooks that explore hyperparameter optimization for different optimizers using Optuna:
- Adam optimizer (`Optuna_Adam.ipynb`)
- AdamW optimizer (`Optuna_AdamW.ipynb`)
- RMSProp optimizer (`Optuna_rmsPROP.ipynb`)

## Project Overview

The project aims to compare the performance of Adam, AdamW, and RMSProp optimizers in training a neural network on the KMNIST dataset, a variation of the MNIST dataset for Japanese characters. Hyperparameter tuning was performed using Optuna to optimize key parameters for each optimizer.

### Notebooks
- **Optuna_Adam.ipynb**: Contains experiments with the Adam optimizer.
- **Optuna_AdamW.ipynb**: Contains experiments with the AdamW optimizer.
- **Optuna_rmsPROP.ipynb**: Contains experiments with the RMSProp optimizer.

### Instructions

1. Clone this repository.
2. Install dependencies (e.g., Optuna, TensorFlow, or PyTorch, depending on the implementation).
3. Open the notebooks in Jupyter and run the cells.

### Dependencies

- Python 3.x
- Jupyter Notebook
- Optuna
- TensorFlow or PyTorch
- KMNIST dataset (included in the code, no additional download required)

## How to Use

1. Install Jupyter: `pip install notebook`
2. Install required libraries: `pip install optuna tensorflow` or `pip install optuna torch`
3. Run Jupyter: `jupyter notebook`
4. Open the notebooks under the `notebooks/` directory and execute the cells.

## Report

For a detailed analysis and results, please refer to the [report.md](report.md) file.
