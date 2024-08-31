from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def load_and_split_cls_data(test_size=0.2, random_state=42):
    X, y = datasets.load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_and_split_reg_data(test_size=0.2, random_state=42):
    machine_cpu = datasets.fetch_openml(name='machine_cpu')
    machine_data = machine_cpu.data
    machine_labels = machine_cpu.target
    return train_test_split(
        machine_data, machine_labels,
        test_size=test_size,
        random_state=random_state)

def train(model, X_train, y_train):
    model = model()
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test, metric):
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)

def main():
    X_train, X_test, y_train, y_test = load_and_split_cls_data()
    dt_classifier = train(DecisionTreeClassifier, X_train, y_train)
    accuracy = evaluate(dt_classifier, X_test, y_test, accuracy_score)
    print(f'Classification Accuracy: {accuracy:.2f}')
    
    X_train, X_test, y_train, y_test = load_and_split_reg_data()
    tree_reg = train(DecisionTreeRegressor, X_train, y_train)
    mse = evaluate(tree_reg, X_test, y_test, mean_squared_error)
    print(f'Mean Squared Error: {mse:.2f}')


if __name__ == '__main__':
    main()
    