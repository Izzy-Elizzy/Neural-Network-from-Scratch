import numpy as np

class Activation_ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    Outputs max(0, input) for each input.
    """
    def forward(self, inputs):
        """
        Apply ReLU activation to inputs.
        
        Args:
            inputs (np.ndarray): Input data
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        """
        Calculate gradient for ReLU activation.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to layer's output
        """
        self.dinputs = dvalues.copy()
        # Set gradient to 0 for negative inputs
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    """
    Softmax activation function for converting raw scores to probabilities.
    """
    def forward(self, inputs):
        """
        Apply softmax activation to inputs.
        
        Args:
            inputs (np.ndarray): Input data
        """
        self.inputs = inputs
        # Subtract max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        """
        Calculate gradient for softmax activation.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to layer's output
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Reshape output to column vector
            single_output = single_output.reshape(-1,1)

            # Compute Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Compute gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)