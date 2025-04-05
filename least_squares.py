from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
import numpy as np
class LeastSquaresClassifier:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.encoder = OneHotEncoder(sparse=False)

    def fit(self, X, y):
        # Convert labels to one-hot format for multi-class regression
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1))
        self.model.fit(X, y_encoded)

    def predict(self, X):
        # Predict gives scores — take argmax across columns
        scores = self.model.predict(X)
        return np.argmax(scores, axis=1)