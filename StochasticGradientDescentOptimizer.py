import numpy as np

class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, learning_rate=0.5):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate (float, optional): Step size for weight updates. Defaults to 0.5.
        """
        self.learning_rate = learning_rate
    
    def update_parameters(self, layer):
        """
        Update layer weights and biases.
        
        Args:
            layer (Layer_Dense): Neural network layer to update
        """
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases