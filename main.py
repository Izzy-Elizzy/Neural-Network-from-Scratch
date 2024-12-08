import numpy as np
import os
from datetime import datetime
import json

# Import the classes we just created
from DenseLayers import Layer_Dense
from ActivationLayers import Activation_ReLU
from CategoricalCrossEntropy import Activation_Softmax_Loss_CategoricalCrossentropy
from StochasticGradientDescentOptimizer import Optimizer_SGD
from Dataset import load_optdigits

class TrainingAnalytics:
    """
    Class to track and store training analytics.
    """
    def __init__(self, epochs):
        """
        Initialize training analytics tracker.
        
        Args:
            epochs (int): Total number of training epochs
        """
        self.epochs = epochs
        self.training_history = {
            'epoch': [],
            'accuracy': [],
            'loss': [],
            'learning_rate': []
        }
        self.test_results = {}
        
    def record_epoch(self, epoch, accuracy, loss, learning_rate):
        """
        Record metrics for a single training epoch.
        
        Args:
            epoch (int): Current epoch number
            accuracy (float): Training accuracy
            loss (float): Training loss
            learning_rate (float): Current learning rate
        """
        self.training_history['epoch'].append(epoch)
        self.training_history['accuracy'].append(accuracy)
        self.training_history['loss'].append(loss)
        self.training_history['learning_rate'].append(learning_rate)
    
    def save_results(self, test_accuracy, test_loss, output_dir='results'):
        """
        Save training and test results to files.
        
        Args:
            test_accuracy (float): Test set accuracy
            test_loss (float): Test set loss
            output_dir (str, optional): Directory to save results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training history
        train_history_path = os.path.join(output_dir, f'training_history_{timestamp}.json')
        with open(train_history_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        
        # Save test results
        test_results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'timestamp': timestamp
        }
        test_results_path = os.path.join(output_dir, f'test_results_{timestamp}.json')
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Generate a summary report
        report_path = os.path.join(output_dir, f'training_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("Neural Network Training Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n\n")
            
            f.write("Training Progression:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Epochs: {self.epochs}\n")
            f.write(f"Final Training Accuracy: {self.training_history['accuracy'][-1]:.4f}\n")
            f.write(f"Final Training Loss: {self.training_history['loss'][-1]:.4f}\n")

def calculate_confusion_matrix(predictions, true_labels, num_classes=10):
    """
    Calculate confusion matrix for multi-class classification.
    
    Args:
        predictions (np.ndarray): Predicted labels
        true_labels (np.ndarray): Ground truth labels
        num_classes (int, optional): Number of classes
    
    Returns:
        np.ndarray: Confusion matrix
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(predictions, true_labels):
        confusion_matrix[true, pred] += 1
    return confusion_matrix

def main():
    """
    Main training and evaluation function for the neural network.
    """
    # Hyperparameters
    EPOCHS = 1000
    LEARNING_RATE = 0.5
    PRINT_INTERVAL = 50  # Print stats every 50 epochs

    # Load MNIST-like handwritten digit dataset
    X_train, X_test, y_train, y_test = load_optdigits()

    # Initialize training analytics
    analytics = TrainingAnalytics(EPOCHS)

    # Create network layers
    dense1 = Layer_Dense(1024, 64)  # First hidden layer 
    activation1 = Activation_ReLU()  # ReLU activation for first layer
    dense2 = Layer_Dense(64, 10)  # Output layer with 10 classes
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Create optimizer
    optimizer = Optimizer_SGD(learning_rate=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS): 
        # Forward pass
        dense1.forward(X_train) 
        activation1.forward(dense1.output) 
        dense2.forward(activation1.output) 
        loss = loss_activation.forward(dense2.output, y_train)

        # Calculate accuracy 
        predictions = np.argmax(loss_activation.output, axis=1) 
        if len(y_train.shape) == 2: 
            y_train_labels = np.argmax(y_train, axis=1) 
        else:
            y_train_labels = y_train
        
        accuracy = np.mean(predictions == y_train_labels) 
        
        # Record epoch metrics
        analytics.record_epoch(
            epoch, 
            accuracy, 
            loss, 
            optimizer.learning_rate
        )
        
        # Print progress at specified intervals
        if not epoch % PRINT_INTERVAL: 
            print(f'Epoch: {epoch}, ' + 
                  f'Accuracy: {accuracy:.4f}, ' + 
                  f'Loss: {loss:.4f}, ' +
                  f'Learning Rate: {optimizer.learning_rate:.4f}') 
        
        # Backward pass 
        loss_activation.backward(loss_activation.output, y_train) 
        dense2.backward(loss_activation.dinputs) 
        activation1.backward(dense2.dinputs) 
        dense1.backward(activation1.dinputs) 
        
        # Update weights and biases 
        optimizer.update_parameters(dense1) 
        optimizer.update_parameters(dense2)

    # Evaluation on test set
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_activation.forward(dense2.output, y_test)

    # Prepare test labels
    test_predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    # Calculate test metrics
    test_accuracy = np.mean(test_predictions == y_test_labels)
    test_loss = loss_activation.loss.calculate(loss_activation.output, y_test)

    # Calculate and save confusion matrix
    confusion_matrix = calculate_confusion_matrix(test_predictions, y_test_labels)
    
    # Save results
    analytics.save_results(test_accuracy, test_loss)

    # Print comprehensive test results
    print('\nComprehensive Test Results')
    print('=' * 30)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Loss: {test_loss:.4f}')
    
    # Print detailed class-wise performance
    print('\nClass-wise Performance:')
    print('-' * 30)
    for cls in range(10):
        class_mask = y_test_labels == cls
        class_accuracy = np.mean(test_predictions[class_mask] == y_test_labels[class_mask])
        class_samples = np.sum(class_mask)
        print(f'Class {cls}: Accuracy = {class_accuracy:.4f}, Samples = {class_samples}')

    # Optionally, save confusion matrix
    np.savetxt('results/confusion_matrix.csv', confusion_matrix, delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()