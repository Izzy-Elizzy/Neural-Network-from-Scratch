import numpy as np
from data import load_optdigits

X = [[1,2,3,2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #Handling for Scalars
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true): 
        # Number of samples 
        samples = len(dvalues) 
        # Number of labels in every sample 
        # We'll use the first sample to count them 
        labels = len(dvalues[0]) 
        # If labels are sparse, turn them into one-hot vector 
        if len(y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true] 
        # Calculate gradient 
        self.dinputs = -y_true / dvalues 
        # Normalize gradient 
        self.dinputs = self.dinputs / samples 
    
class Activation_Softmax_Loss_CategoricalCrossentropy(): 
# Creates activation and loss function objects 
    def __init__(self): 
        self.activation = Activation_Softmax() 
        self.loss = Loss_CategoricalCrossEntropy()  
        # Forward pass 
    def forward(self, inputs, y_true): 
    # Output layer's activation function 
        self.activation.forward(inputs) 
        # Set the output 
        self.output = self.activation.output 
        # Calculate and return loss value 
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true): 
        # Number of samples 
        samples = len(dvalues) 
        # If labels are one-hot encoded, 
        # turn them into discrete values 
        if len(y_true.shape) == 2: 
            y_true = np.argmax(y_true, axis=1) 
        # Copy so we can safely modify 
        self.dinputs = dvalues.copy() 
        # Calculate gradient 
        self.dinputs[range(samples), y_true] -= 1 
        # Normalize gradient 
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
    def update_parameters(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

X_train, X_test, y_train, y_test = load_optdigits()

dense1 = Layer_Dense(1024, 64) 
# Create ReLU activation (to be used with Dense layer): 
activation1 = Activation_ReLU() 
# Create second Dense layer with 64 input features (as we take output 
# of previous layer here) and 3 output values (output values) 
dense2 = Layer_Dense(64, 10) 
# Create Softmax classifier's combined loss and activation 
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD()

for epoch in range(2000): 
    # Perform a forward pass of our training data through this layer 
    dense1.forward(X_train) 
    # Perform a forward pass through activation function 
    # takes the output of first dense layer here 
    activation1.forward(dense1.output) 
    # Perform a forward pass through second Dense layer 
    # takes outputs of activation function of first layer as inputs 
    dense2.forward(activation1.output) 
    # Perform a forward pass through the activation/loss function 
    # takes the output of second dense layer here and returns loss 
    loss = loss_activation.forward(dense2.output, y_train)

    # Calculate accuracy from output of activation2 and targets 
    # calculate values along first axis 
    predictions = np.argmax(loss_activation.output, axis=1) 
    if len(y_train.shape) == 2: 
        y_train = np.argmax(y_train, axis=1) 
    accuracy = np.mean(predictions==y_train) 
    if not epoch % 100: 
        print(f'epoch: {epoch}, ' + 
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {loss:.3f}') 
    # Backward pass 
    loss_activation.backward(loss_activation.output, y_train) 
    dense2.backward(loss_activation.dinputs) 
    activation1.backward(dense2.dinputs) 
    dense1.backward(activation1.dinputs) 
    # Update weights and biases 
    optimizer.update_parameters(dense1) 
    optimizer.update_parameters(dense2)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_activation.forward(dense2.output, y_test)

test_predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_test_labels= y_test

test_accuracy = np.mean(test_predictions == y_test_labels)
test_loss = loss_activation.loss.calculate(loss_activation.output, y_test)

print('\nTest Results')
print(f'\nTest Accuracy: {test_accuracy:.3f}')
print(f'\nTest Loss: {test_loss:.3f}')

# print(X_train[0])

# dense1 = Layer_Dense(1024, 32)
# activation1 = Activation_ReLU()

# dense2 = Layer_Dense(32, 10)
# activation2 = Activation_Softmax()

# dense1.forward(X_train)
# activation1.forward(dense1.output)

# dense2.forward(activation1.output)
# activation2.forward(dense2.output)


# loss_function = Loss_CategoricalCrossEntropy()

# lowest_loss = 9999999
# best_dense1_weights = dense1.weights.copy()
# best_dense1_biases = dense1.biases.copy()
# best_dense2_weights = dense2.weights.copy()
# best_dense2_biases = dense2.biases.copy()

# for iteration in range(10000):
#     dense1.weights += 0.05 * np.random.randn(1024,32)
#     dense1.biases += 0.05 * np.random.randn(1,32)
#     dense2.weights += 0.05 * np.random.randn(32,10)
#     dense2.biases += 0.05 * np.random.randn(1,10)

#     dense1.forward(X_train)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)

#     loss = loss_function.calculate(activation2.output, y_train)

#     if loss < lowest_loss:
#         print('New Set of Weights and Biases Found, Iteration: ', iteration,
#               'loss: ', loss)    
#         best_dense1_weights = dense1.weights.copy()
#         best_dense1_biases = dense1.biases.copy()
#         best_dense2_weights = dense2.weights.copy()
#         best_dense2_biases = dense2.biases.copy()
#         lowest_loss = loss
    
#     else:
#         dense1.weights = best_dense1_weights.copy()
#         dense1.biases = best_dense1_biases.copy()
#         dense2.weights = best_dense2_weights.copy()
#         dense2.biases = best_dense2_biases.copy()

# weights1 = [[0.2, 0.8, -0.5, 1.0],
#             [0.5, -0.91, 0.26, -0.5],
#             [-0.26, -0.27, 0.17, 0.87]]

# biases1 = [2, 3, 0.5]

# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

# inputs = [1,2,3,2.5]
# weights = [[0.2, 0.8, -0.5, 1.0],
#             [0.5, -0.91, 0.26, -0.5],
#             [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights,biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs,neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

# print(layer_outputs)

