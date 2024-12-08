# Neural Network from Scratch: Digit Classification

## Project Overview

This project implements a neural network from scratch using NumPy to classify handwritten digits from the MNIST-like optdigits dataset. The implementation demonstrates core machine learning concepts including forward propagation, backpropagation, activation functions, and optimization techniques.

## Features

- **Custom Neural Network Implementation**
  - Fully connected dense layers
  - ReLU and Softmax activation functions
  - Categorical Cross-Entropy loss
  - Stochastic Gradient Descent (SGD) optimizer

- **Data Processing**
  - Digit image dataset loading
  - Automatic train-test splitting
  - 1024-feature input representation

- **Training Analytics**
  - Epoch-wise accuracy and loss tracking
  - Automatic result logging
  - Confusion matrix generation
  - Detailed class-wise performance reporting

## Project Structure

```
neural-network-from-scratch/
│
├── Dataset.py         # Data loading and preprocessing
├── DenseLayers.py     # Dense layer implementation
├── ActivationLayers.py # Activation function layers
├── LossFunctions.py   # Loss function implementations
├── main.py            # Primary training and evaluation script
└── StochasticGradientDescentOptimizer.py # SGD optimizer
```

## Prerequisites

- Python 3.8+
- NumPy
- JSON (standard library)

## Hyperparameters

- **Network Architecture**: 
  - Input Layer: 1024 features
  - Hidden Layer: 64 neurons with ReLU activation
  - Output Layer: 10 neurons with Softmax activation

- **Training Configuration**:
  - Epochs: 1000
  - Learning Rate: 0.5
  - Optimizer: Stochastic Gradient Descent (SGD)

## Usage

```bash
python main.py
```

## Output

The script generates the following outputs in the `results/` directory:
- Training history (JSON)
- Test results (JSON)
- Training report (TXT)
- Confusion matrix (CSV)

## Performance Metrics

The script provides:
- Overall Test Accuracy
- Test Loss
- Class-wise Accuracy
- Confusion Matrix

## Key Concepts Demonstrated

1. Neural network architecture design
2. Forward and backward propagation
3. Activation functions (ReLU, Softmax)
4. Loss calculation
5. Gradient-based optimization
6. Model evaluation techniques

## Limitations

- Fixed neural network architecture
- No dynamic hyperparameter tuning
- Single dataset (optdigits)

## Acknowledgments

- Inspired by machine learning from-scratch implementations
- MNIST-like optdigits dataset