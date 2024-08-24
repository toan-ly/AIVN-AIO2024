# !pip install -q -U datasets
import numpy as np
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Load IMDB dataset
imdb = load_dataset('imdb')
imdb_train, imdb_test = imdb['train'], imdb['test']

# Convert text to vector with BoW
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(imdb_train['text']).toarray()
X_test = vectorizer.transform(imdb_test['text']).toarray()
y_train = imdb_train['label']
y_test = imdb_test['label']

# Scale the features
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build KNN Classifier
knn_cls = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
knn_cls.fit(X_train, y_train)

# Predict and Evaluate
y_pred = knn_cls.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy of KNN Text Classifier: {acc:.2f}')