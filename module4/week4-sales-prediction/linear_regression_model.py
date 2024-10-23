import numpy as np

class CustomLinearRegression:
    def __init__(self, X, y, lr=0.01, epochs=10000):
        self.num_samples = X.shape[0]
        self.X = np.c_[np.ones((self.num_samples, 1)), X]
        self.y = y
        self.lr = lr
        self.epochs = epochs

        # Initialize weights (theta) randomly
        self.theta = np.random.randn(self.X.shape[1], 1)
        self.losses = []

    def compute_loss(self, y_pred, y):
        # print(y_pred.shape, y.shape) # For debugging
        return np.mean((y_pred - y) ** 2)
    
    def predict(self, X):
        try:
            y_pred = X.dot(self.theta)
        except:
            X = np.c_[np.ones((X.shape[0], 1)), X]
            y_pred = X.dot(self.theta)
        return y_pred

    def fit(self):
        for epoch in range(self.epochs):
            y_pred = self.predict(self.X)

            # Compute loss
            loss = self.compute_loss(y_pred, self.y)
            self.losses.append(loss)

            # Compute gradients
            grad = self.X.T.dot(2 * (y_pred - self.y)) / self.num_samples

            # Update weights
            self.theta -= self.lr * grad
            
            if epoch % 50 == 0:
                print(f"Epoch: {epoch} - Loss: {loss}")

        return {'loss': sum(self.losses) / len(self.losses), 'weight': self.theta}
