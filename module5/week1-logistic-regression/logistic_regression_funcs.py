import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(np.dot(X, theta))

def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1-1e-7)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def compute_gradient(X, y, y_hat):
    return np.dot(X.T, y_hat - y) / y.size

def update_theta(theta, gradient, lr):
    return theta - lr * gradient

def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta).round()
    return (y_hat == y).mean()