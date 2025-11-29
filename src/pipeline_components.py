"""Pipeline components for data extraction, preprocessing, training, and evaluation."""

import os
import subprocess
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils import load_csv, save_csv, save_numpy_array, compute_metrics, save_json


def extract_data(dvc_path: str, local_path: str) -> str:
    """Extract data using DVC pull.
    
    Args:
        dvc_path: Path to the DVC file (e.g., "data.dvc")
        local_path: Local path where data will be stored
        
    Returns:
        Path to the extracted raw CSV file
    """
    # Check if file already exists locally
    if os.path.exists(local_path):
        print(f"Data file already exists at {local_path}, skipping DVC pull")
        mlflow.log_param("data_source", "local_file")
        mlflow.log_artifact(local_path, "raw_data")
        return local_path
    
    # Try to run dvc pull to get the data
    try:
        result = subprocess.run(
            ['dvc', 'pull', dvc_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"DVC pull successful: {result.stdout}")
        mlflow.log_param("dvc_path", dvc_path)
        mlflow.log_param("data_source", "dvc_pull")
        mlflow.log_param("data_extracted", True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: DVC pull failed: {e.stderr if hasattr(e, 'stderr') else str(e)}")
        print("Attempting to continue with local file if it exists...")
    except FileNotFoundError:
        print("DVC not found in PATH, skipping DVC pull")
        print("Attempting to use local file if it exists...")
    
    # Check if file exists after DVC pull attempt
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Data file not found at {local_path}. "
            f"Please ensure the file exists locally or set up DVC remote storage."
        )
    
    try:
        mlflow.log_artifact(local_path, "raw_data")
    except Exception as e:
        print(f"Warning: Failed to log artifact to MLflow: {e}")
    return local_path


def preprocess_data(
    input_csv: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """Preprocess data and create train/test split.
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save processed data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to train/test data files
    """
    # Load data
    df = load_csv(input_csv)
    
    # Separate features and target
    # Assuming MEDV is the target column (last column)
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    train_X_path = os.path.join(output_dir, 'X_train.npy')
    train_y_path = os.path.join(output_dir, 'y_train.npy')
    test_X_path = os.path.join(output_dir, 'X_test.npy')
    test_y_path = os.path.join(output_dir, 'y_test.npy')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    save_numpy_array(X_train_scaled, train_X_path)
    save_numpy_array(y_train.values, train_y_path)
    save_numpy_array(X_test_scaled, test_X_path)
    save_numpy_array(y_test.values, test_y_path)
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    
    # Log preprocessing parameters and artifacts
    try:
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_artifact(scaler_path, "preprocessing")
    except Exception as e:
        print(f"Warning: Failed to log preprocessing to MLflow: {e}")
    
    return {
        'X_train': train_X_path,
        'y_train': train_y_path,
        'X_test': test_X_path,
        'y_test': test_y_path,
        'scaler': scaler_path
    }


def train_model(
    train_data_paths: dict,
    model_output_path: str,
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42
) -> str:
    """Train a RandomForestRegressor model.
    
    Args:
        train_data_paths: Dictionary with paths to X_train and y_train
        model_output_path: Directory to save the trained model
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Model URI for MLflow
    """
    # Load training data
    X_train = np.load(train_data_paths['X_train'])
    y_train = np.load(train_data_paths['y_train'])
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Save model locally first
    os.makedirs(model_output_path, exist_ok=True)
    model_path = os.path.join(model_output_path, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved locally to: {model_path}")
    
    # Log model parameters and to MLflow
    try:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestRegressor")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        model_uri = mlflow.get_artifact_uri("model")
        
        # Log model file as artifact
        mlflow.log_artifact(model_path, "models")
        print(f"Model logged to MLflow: {model_uri}")
    except Exception as e:
        print(f"Warning: Failed to log model to MLflow: {e}")
        # Return a local URI if MLflow fails
        model_uri = f"file://{os.path.abspath(model_path)}"
    
    return model_uri


def evaluate_model(
    model_uri: str,
    test_data_paths: dict,
    metrics_output_path: str
) -> dict:
    """Evaluate a trained model and compute metrics.
    
    Args:
        model_uri: MLflow model URI
        test_data_paths: Dictionary with paths to X_test and y_test
        metrics_output_path: Path to save metrics JSON file
        
    Returns:
        Dictionary containing computed metrics
    """
    # Load test data
    X_test = np.load(test_data_paths['X_test'])
    y_test = np.load(test_data_paths['y_test'])
    
    # Load model from MLflow
    model = mlflow.sklearn.load_model(model_uri)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    
    # Save metrics to JSON
    save_json(metrics, metrics_output_path)
    print(f"Metrics saved to: {metrics_output_path}")
    
    # Log metrics to MLflow
    try:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_output_path, "metrics")
        print("Metrics logged to MLflow successfully")
    except Exception as e:
        print(f"Warning: Failed to log metrics to MLflow: {e}")
    
    return metrics

