"""Utility functions for data loading, saving, and metrics computation."""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    return pd.read_csv(file_path)


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """Save a pandas DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path where to save the CSV file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)


def save_numpy_array(arr: np.ndarray, file_path: str) -> None:
    """Save a numpy array to a file.
    
    Args:
        arr: Numpy array to save
        file_path: Path where to save the array
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, arr)


def load_numpy_array(file_path: str) -> np.ndarray:
    """Load a numpy array from a file.
    
    Args:
        file_path: Path to the numpy array file
        
    Returns:
        Loaded numpy array
    """
    return np.load(file_path)


def save_json(data: dict, file_path: str) -> None:
    """Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str) -> dict:
    """Load a JSON file into a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing MSE, RMSE, MAE, and R² metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }

