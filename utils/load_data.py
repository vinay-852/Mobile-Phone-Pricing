import os
import pandas as pd
def load_data(data_path):
    """
    Load the dataset from the specified path.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    data = pd.read_csv(data_path)
    X = data.drop(columns=['price_range'])
    y = data['price_range']
    
    return X, y