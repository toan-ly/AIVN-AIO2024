import numpy as np
from logistic_regression_funcs import *

def train(X_train, y_train, X_val, y_val, lr=0.01, epochs=100, batch_size=16):
    theta = np.random.uniform(size=X_train.shape[1])
    train_acc, train_losses, val_acc, val_losses = [], [], [], []
    
    for epoch in range(epochs):
        train_batch_losses, train_batch_acc, val_batch_losses, val_batch_acc = [], [], [], []

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_hat = predict(X_batch, theta)
            loss = compute_loss(y_hat, y_batch)
            gradient = compute_gradient(X_batch, y_batch, y_hat)
            theta = update_theta(theta, gradient, lr)

            train_batch_losses.append(loss)
            train_batch_acc.append(compute_accuracy(X_train, y_train, theta))

            y_val_hat = predict(X_val, theta)
            val_loss = compute_loss(y_val_hat, y_val)
            val_batch_losses.append(val_loss)
            val_batch_acc.append(compute_accuracy(X_val, y_val, theta))

        train_losses.append(np.mean(train_batch_losses))
        train_acc.append(np.mean(train_batch_acc))
        val_losses.append(np.mean(val_batch_losses))
        val_acc.append(np.mean(val_batch_acc))
        
        print(f'Epoch {epoch+1}/{epochs}, train_loss: {train_losses[-1]:.3f}, train_acc: {train_acc[-1]:.3f}, val_loss: {val_losses[-1]:.3f}, val_acc: {val_acc[-1]:.3f}')
    
    return theta, train_acc, train_losses, val_acc, val_losses