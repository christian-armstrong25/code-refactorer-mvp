from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X_new):
    predictions = model.predict(X_new)
    return predictions

def evaluate_model(model, X_test, y_test):
    score = model.score(X_test, y_test)
    return score
