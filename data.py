import numpy as np

def load_optdigits(test_size=0.2, random_state=42):
    """
    Load digits from the optdigits-orig.windep file and split into training and test sets.
    
    Parameters:
    -----------
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before applying the split
    
    Returns:
    --------
    X_train : numpy array
        Training digit features
    X_test : numpy array
        Test digit features
    y_train : numpy array
        Training labels
    y_test : numpy array
        Test labels
    """
    # Load digits and labels
    with open('./data/optdigits-orig.windep') as file:
        content = file.readlines()

    digits = []
    labels = []
    start_offset = 21
    digit_offset = 33

    # Load all digits
    while start_offset + digit_offset <= len(content):
        # Get the digit lines
        digit = []
        for line in content[start_offset:start_offset + digit_offset - 1]:  # Exclude the last line (label)
            digit.extend([int(x) for x in line.strip()])
        
        # Get the label
        label = int(content[start_offset + digit_offset - 1].strip())
        
        # Append to our lists
        digits.append(digit)
        labels.append(label)
        
        # Update start_offset for next iteration
        start_offset += digit_offset

    # Convert to numpy arrays
    X = np.array(digits)
    y = np.array(labels)

    # Ensure the shape is correct (1024 features)
    assert X.shape[1] == 1024, f"Unexpected feature shape: {X.shape}"

    # Shuffle the data
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Calculate split index
    split_index = int(len(X) * (1 - test_size))

    # Split the data
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

# Example usage
X_train, X_test, y_train, y_test = load_optdigits()

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)