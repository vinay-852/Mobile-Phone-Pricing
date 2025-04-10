from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os 
class Model:
    def __init__(self):
        self.scalar = StandardScaler()
        self.model = LogisticRegression()

    def train(self, X, y):
        """
        Train the model with the provided training data (X) and labels (y).
        It scales the data before fitting the logistic regression model.
        """
        X_scaled = self.scalar.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        """
        Predict the class labels for the given input data (X).
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been trained yet.")
        X_scaled = self.scalar.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the given input data (X).
        """
        X_scaled = self.scalar.transform(X)
        return self.model.predict_proba(X_scaled)

    def load(self, model_path, scalar_path):
        """
        Load a pre-trained model and scalar from disk.
        """
        self.model = joblib.load(model_path)
        self.scalar = joblib.load(scalar_path)

    def save(self, path):
        """
        Save the model and scalar to the given path.
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, f"{path}/model.pkl")
        joblib.dump(self.scalar, f"{path}/scalar.pkl")

    def is_trained(self):
        """
        Check if the model has been trained.
        """
        return hasattr(self, 'model') and self.model is not None