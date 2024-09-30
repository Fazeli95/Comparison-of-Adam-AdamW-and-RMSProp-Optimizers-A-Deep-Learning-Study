# Comparison of Adam, AdamW, and RMSProp Optimizers: A Deep Learning Study

## Introduction
In this project, I compared the performance of three popular optimizers—Adam, AdamW, and RMSProp—using Optuna for hyperparameter tuning. The experiment was conducted on the **KMNIST dataset**, which is a variation of the MNIST dataset, featuring Japanese characters. Optimizers are crucial for training neural networks, and this project aimed to highlight their differences in terms of speed, convergence, and generalization.

## Objective
The aim was to compare:
- Convergence speed and stability.
- Generalization ability (validation accuracy).
- The impact of hyperparameter optimization on each optimizer.

## Experimentation with Default Hyperparameters
For each optimizer, we experimented around the default values for key hyperparameters such as the learning rate, beta values, and weight decay. This allowed us to explore the optimal ranges for each optimizer's hyperparameters, which can differ based on the optimizer's design. For example:
- Adam and AdamW have default learning rates of `1e-3`, while RMSProp defaults to `1e-2`. 
- Similarly, Adam's beta values default to `(0.9, 0.999)`, while RMSProp does not use betas but instead relies on a momentum value.
  
This approach of starting near the default values is considered best practice as it ensures that the optimizer is tested within its effective range, preventing drastic deviations that could affect performance unfairly.

## Optimizers Overview

1. **Adam**: Known for fast convergence, Adam uses running averages of gradients and second moments of gradients.
2. **AdamW**: An enhanced version of Adam, decoupling weight decay from gradient updates, thus improving regularization.
3. **RMSProp**: Designed to adjust learning rates based on the average of recent gradients, providing stability in training.

## Methodology

1. **Dataset**: The **KMNIST dataset** was used for this project. It consists of 70,000 28x28 grayscale images of 10 classes of Japanese characters.
2. **Model**: A simple feed-forward neural network was trained using each optimizer, with hyperparameter tuning performed via Optuna.
3. **Hyperparameter Optimization**: Optuna was used to optimize the learning rate, momentum, and weight decay for Adam, AdamW, and RMSProp.

## Results

1. **Adam**:
   - **Best Validation Accuracy**: 90.73%
   - **Training Loss (Final Epoch)**: 1.5535
   - **Validation Loss (Final Epoch)**: 1.5826
   - **Observation**: Adam converged quickly, achieving the highest validation accuracy. However, it showed slight overfitting in the final epochs.

2. **AdamW**:
   - **Best Validation Accuracy**: 89.94%
   - **Training Loss (Final Epoch)**: 1.5729
   - **Validation Loss (Final Epoch)**: 1.5872
   - **Observation**: AdamW performed nearly as well as Adam but had better regularization, preventing overfitting.

3. **RMSProp**:
   - **Best Validation Accuracy**: 87.87%
   - **Training Loss (Final Epoch)**: 1.5765
   - **Validation Loss (Final Epoch)**: 1.5949
   - **Observation**: RMSProp converged more slowly, underperforming compared to Adam and AdamW. This may be due to the limited number of epochs in the experiment.

## Limitations

Due to computational constraints, each optimizer was only trained for 9 epochs per fold. RMSProp, which typically converges more slowly, may have achieved better results given more training cycles.

## Conclusion
- **Adam**: Best for fast convergence and high accuracy, though prone to overfitting.
- **AdamW**: Provides a better balance between convergence and regularization, making it a strong choice for tasks requiring good generalization.
- **RMSProp**: Slower convergence and lower accuracy in this experiment, likely due to insufficient training epochs.

For future experiments, allowing more epochs and computational power could reveal better performance for RMSProp.
