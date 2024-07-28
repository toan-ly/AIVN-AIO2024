import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    def __init__(self):
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        labels = np.unique(y)
        for label in labels:
            class_data = X[y == label]
            self.mean[label] = np.mean(class_data, axis=0)
            self.var[label] = np.var(class_data, axis=0)
            self.priors[label] = class_data.shape[0] / X.shape[0]

    def _gaussian(self, x, class_idx):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-(x - mean) ** 2 / (2 * var))

    def predict(self, X):
        posteriors = []
        for x in X:
            class_posteriors = []
            for label in self.mean:
                # Calculate the log prior probability
                prior = np.log(self.priors[label])
                # Caculate the log of the conditional probabilities
                conditional = np.sum(np.log(self._gaussian(x, label)))
                # Sum the prior and conditional probabilities to get the posterior
                posterior = prior + conditional
                class_posteriors.append(posterior)

            # Determine class with highest posterior
            best_class = list(self.mean.keys())[np.argmax(class_posteriors)]
            posteriors.append(best_class)
        return posteriors
            
            
def load_iris_data(filepath):
    data = pd.read_csv(filepath, delimiter=',')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def main():
    iris_data = np.array([
        [1.4, 0],
        [1.0, 0],
        [1.3, 0],
        [1.9, 0],
        [2.0, 0],
        [1.8, 0],
        [3.0, 1],
        [3.8, 1],
        [4.1, 1],
        [3.9, 1],
        [4.2, 1],
        [3.4, 1]
    ])

    labels = np.unique(iris_data[:, 1])
    for label in labels:
        class_data = iris_data[iris_data[:, 1] == label]
        mean = np.mean(class_data[:, 0])
        var = np.var(class_data[:, 0]
                    )
        print(f'Class: {int(label)}, mean = {mean:.3f}, variance = {var:.3f}')

    X, y = load_iris_data('iris_data.txt')
    gauss_naive_bayes = GaussianNaiveBayes()
    gauss_naive_bayes.fit(X, y)

 
    # Example 1
    X_test = np.array([[6.3, 3.3, 6.0, 2.5]])
    pred = gauss_naive_bayes.predict(X_test)
    print(f'Prediction for {X_test}: {pred[0]}')
    assert pred[0] == "Iris-virginica"

    # Example 2
    X_test = np.array([[5.0, 2.0, 3.5, 1.0]])
    pred = gauss_naive_bayes.predict(X_test)
    print(f'Prediction for {X_test}: {pred[0]}')
    assert pred[0] == "Iris-versicolor"

    # Example 3
    X_test = np.array([[4.9, 3.1, 1.5, 0.1]])
    pred = gauss_naive_bayes.predict(X_test)
    print(f'Prediction for {X_test}: {pred[0]}')
    assert pred[0] == "Iris-setosa"

if __name__ == "__main__":
    main()