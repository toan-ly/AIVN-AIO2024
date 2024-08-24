import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

from kmeans import KMeans

def test_iris():
    # Load the Iris dataset
    iris_dataset = load_iris()
    data = iris_dataset.data[:, :2]

    # Plot data
    plt.scatter(data[:, 0], data[:, 1], c='gray')
    plt.title('Initial Dataset')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()

    # Apply KMeans with k = 3
    kmeans = KMeans(k=3)
    kmeans.fit(data)
    kmeans = KMeans(k=4)
    kmeans.fit(data)

def test_custom():
    data = np.array([
        [2.0, 3.0, 1.5],
        [3.0, 3.5, 2.0],
        [3.5, 3.0, 2.5],
        [8.0, 8.0, 7.5],
        [8.5, 8.5, 8.0],
        [9.0, 8.0, 8.5],
        [1.0, 2.0, 1.0],
        [1.5, 2.5, 1.5],
    ])
    df = pd.DataFrame(data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    print(df)
    print(data.shape)

    kmeans = KMeans(k=2)
    centroids = [data[0], data[6]]
    print(data[0], data[3])
    print(round(kmeans.euclidean_distance(data[0], data[3])))

    sample = data[1]
    print(np.argmin(kmeans.euclidean_distance(sample, centroids)))

    kmeans = KMeans(k=3)
    kmeans.fit(data)
    print(kmeans.centroids)


def main():
    # test_iris()
    test_custom()


if __name__ == '__main__':
    main()