from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def evaluate_model(regressor, X, y):
    y_pred = regressor.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print('Evaluation results on validation set:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

def plot(X, feature, y, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, feature], y, color='blue', label='Ground Truth', s=60, edgecolor='k', alpha=0.6)
    plt.scatter(X[:, feature], y_pred, color='green', label='Prediction', s=60, edgecolor='k', alpha=0.6)

    plt.title(f'{model_name} - {feature} vs Predicted and Actual', fontsize=18, weight='bold', color='#333')
    plt.xlabel(f'{feature}', fontsize=14, weight='bold')
    plt.ylabel('Price', fontsize=14, weight='bold')

    plt.xticks(fontsize=12, weight='bold')
    plt.yticks(fontsize=12, weight='bold')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()