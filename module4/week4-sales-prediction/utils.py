import  numpy as np

def r2_score(y_pred, y):
    rss = np.sum((y_pred - y) ** 2) # Residual sum of squares
    tss = np.sum((np.mean(y) - y) ** 2) # Total sum of squares
    r2 = 1 - rss / tss
    return r2

def create_polynomial_features(X, degree=2):
    X_mem = []
    for X_sub in X.T:
        X_sub = X_sub.T
        X_new = X_sub
        for d in range(2, degree + 1):
            X_new = np.c_[X_new, np.power(X_sub, d)]
        X_mem.extend(X_new.T)
    return np.c_[X_mem].T
