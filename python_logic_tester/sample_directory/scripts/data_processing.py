import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    data = data.dropna()
    data = data[data['value'] >= 0]
    return data

def save_clean_data(data, filepath):
    data.to_csv(filepath, index=False)
