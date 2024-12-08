import numpy as np

class Loss:
    """
    Base class for loss functions.
    """
    def calculate(self, output, y):
        """
        Calculate mean loss across all samples.
        
        Args:
            output (np.ndarray): Network output
            y (np.ndarray): True labels
        
        Returns:
            float: Mean loss
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss function for multi-class classification.
    """
    def forward(self, y_pred, y_true):
        """
        Calculate loss for each sample.
        
        Args:
            y_pred (np.ndarray): Predicted probabilities
            y_true (np.ndarray): True labels
        
        Returns:
            np.ndarray: Loss for each sample
        """
        samples = len(y_pred)
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Handle different label format (scalar or one-hot)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        # Compute negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true): 
        """
        Calculate gradient for categorical cross-entropy loss.
        
        Args:
            dvalues (np.ndarray): Output from softmax
            y_true (np.ndarray): True labels
        """
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true] 
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues 
        # Normalize gradient 
        self.dinputs = self.dinputs / samples