from models.model import Model
from utils.load_data import load_data
import os

def predict(model, X):
    """
    Predict the class labels for the given input data (X).
    """
    if not model.is_trained():
        raise ValueError("Model has not been trained yet.")
    
    return model.predict(X)
def predict_proba(model, X):
    """
    Predict the class probabilities for the given input data (X).
    """
    if not model.is_trained():
        raise ValueError("Model has not been trained yet.")
    
    return model.predict_proba(X)
def main():
    model_path = 'models/trained_model'
    data_path = 'data/dataset.csv'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    model = Model()
    model.load(os.path.join(model_path, 'model.pkl'), os.path.join(model_path, 'scalar.pkl'))
    
    X, _ = load_data(data_path)
    
    predictions = predict(model, X)
    probabilities = predict_proba(model, X)
    
    print("Predictions:", predictions)
    print("Probabilities:", probabilities)
if __name__ == "__main__":
    main()