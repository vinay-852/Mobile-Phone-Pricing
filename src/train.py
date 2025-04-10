import os
from models.model import Model
from utils.load_data import load_data

def train_model(model, X, y, model_path):
    """
    Train the model with the provided training data (X) and labels (y).
    Save the trained model and scalar to the specified path.
    """
    model.train(X, y)
    model.save(model_path)
    
    print(f"Model trained and saved to {model_path}")

def main():
    data_path = 'data/dataset.csv'
    model_path = 'models/trained_model'
    
    # Check if the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} does not exist.")
        return  # Exit the program if the file is missing

    X, y = load_data(data_path)
    model = Model()
    train_model(model, X, y, model_path)
    print("Training completed.")

if __name__ == "__main__":
    main()