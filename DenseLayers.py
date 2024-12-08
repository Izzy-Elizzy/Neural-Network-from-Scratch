import numpy as np

class Layer_Dense:
    """
    A dense (fully connected) neural network layer.
    
    Attributes:
        weights (np.ndarray): Layer weights initialized randomly
        biases (np.ndarray): Layer biases initialized to zeros
    """
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values
        # Multiplying by 0.10 helps prevent large initial weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        """
        Perform forward propagation through the layer.
        
        Args:
            inputs (np.ndarray): Input data from previous layer
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    
    def backward(self, dvalues):
        """
        Perform backward propagation to calculate gradients.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to layer's output
        """
        # Gradient for weights: inputs transposed dot product with dvalues
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        # Gradient for biases: sum of dvalues across samples
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient for inputs: dvalues dot product with weights transposed
        self.dinputs = np.dot(dvalues, self.weights.T)