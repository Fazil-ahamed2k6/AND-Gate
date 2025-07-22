
import numpy as np

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Inputs (X) and Expected Outputs (Y)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0
learning_rate = 0.1

# Training the perceptron
for epoch in range(10):  # 10 times over the data
    print(f"Epoch {epoch+1}")
    for i in range(len(X)):
        x = X[i]
        y_true = Y[i]

        # Linear combination
        linear_output = np.dot(x, weights) + bias
        y_pred = step(linear_output)

        # Error
        error = y_true - y_pred

        # Update rule
        weights += learning_rate * error * x
        bias += learning_rate * error

        print(f"Input: {x}, Prediction: {y_pred}, Error: {error}, New weights: {weights}, Bias: {bias}")
    print()

# Testing
print("Final model predictions:")
for x in X:
    y_pred = step(np.dot(x, weights) + bias)
    print(f"Input: {x} => Output: {y_pred}")