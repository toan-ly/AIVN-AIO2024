from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(regressor, X, y):
    y_pred = regressor.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    print('Evaluation results on validation set:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
