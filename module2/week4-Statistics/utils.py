import numpy as np

def compute_mean(X):
    return np.mean(X)

def compute_median(X):
    X = np.sort(X)
    n = len(X)
    print(X)
    if n % 2 == 0:
        return (X[n // 2 - 1] + X[n // 2]) / 2.0
    return X[n // 2]

def compute_std(X):
    mean = compute_mean(X)
    var = np.mean((X - mean) ** 2)
    return np.sqrt(var)

def compute_correlation_coefficient(X, Y):
    n = len(X)
    sum_X = np.sum(X)
    sum_Y = np.sum(Y)
    numerator = n * X.dot(Y) - sum_X * sum_Y
    denominator = np.sqrt((n * np.sum(X**2) - sum_X**2) * (n * np.sum(Y**2) - sum_Y**2))
    return np.round(numerator / denominator, 2)


