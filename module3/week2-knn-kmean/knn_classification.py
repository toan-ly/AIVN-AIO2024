import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)

# Split train/test = 8/2
X_train, X_test, y_train, y_test = train_test_split(
    iris_X,
    iris_y,
    test_size=0.2,
    random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Builld KNN Classifier
knn_cls = KNeighborsClassifier(n_neighbors=5)
knn_cls.fit(X_train, y_train)

# Predict and Evaluate test set
y_pred = knn_cls.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy of KNN Classifier: {acc:.2f}')