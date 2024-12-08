import numpy as np
from ActivationLayers import Activation_Softmax
from LossFunctions import Loss_CategoricalCrossEntropy

class Activation_Softmax_Loss_CategoricalCrossentropy(): 
    """
    Combined Softmax activation and Categorical Cross-Entropy loss.
    Provides forward and backward passes for the output layer.
    """
    def __init__(self): 
        self.activation = Activation_Softmax() 
        self.loss = Loss_CategoricalCrossEntropy()  
    
    def forward(self, inputs, y_true): 
        """
        Perform forward pass with softmax activation and loss calculation.
        
        Args:
            inputs (np.ndarray): Raw network output
            y_true (np.ndarray): True labels
        
        Returns:
            float: Calculated loss
        """
        self.activation.forward(inputs) 
        self.output = self.activation.output 
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true): 
        """
        Perform backward pass for combined softmax and cross-entropy.
        
        Args:
            dvalues (np.ndarray): Output from softmax
            y_true (np.ndarray): True labels
        """
        samples = len(dvalues) 
        
        # Convert one-hot labels to discrete values if needed
        if len(y_true.shape) == 2: 
            y_true = np.argmax(y_true, axis=1) 
        
        # Copy to safely modify 
        self.dinputs = dvalues.copy() 
        
        # Calculate gradient 
        self.dinputs[range(samples), y_true] -= 1 
        
        # Normalize gradient 
        self.dinputs = self.dinputs / samples