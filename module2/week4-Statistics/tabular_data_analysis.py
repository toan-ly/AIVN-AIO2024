import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = (x - x_mean).dot(y - y_mean)
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    if denominator == 0:
        return 0
    return numerator / denominator

def plot_heatmap(data):
    data_corr = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(data_corr, annot=True, fmt='.2f', linewidths=.5)
    plt.show()