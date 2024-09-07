def train_model(model, X, y, random_state=1):
    regressor = model(random_state=random_state)
    regressor.fit(X, y)
    return regressor



    