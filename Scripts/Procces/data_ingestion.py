import pandas as pd
def load_data(filepath):
    """ Load data from a CSV file """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
