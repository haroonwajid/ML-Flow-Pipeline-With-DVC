"""Main pipeline orchestration script using MLflow."""

import argparse
import yaml
import mlflow
from src.pipeline_components import (
    extract_data,
    preprocess_data,
    train_model,
    evaluate_model
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str) -> None:
    """Run the complete ML pipeline.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set MLflow tracking URI (defaults to local file store)
    # Can be overridden with MLFLOW_TRACKING_URI environment variable
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Start MLflow run
    with mlflow.start_run(run_name=config['run_name']):
        # Log configuration
        mlflow.log_params({
            'seed': config['seed'],
            'test_size': config['data']['test_size'],
            'n_estimators': config['model']['n_estimators'],
            'max_depth': config['model']['max_depth']
        })
        
        # Step 1: Extract data
        print("Step 1: Extracting data...")
        raw_data_path = extract_data(
            dvc_path=config['data']['dvc_path'],
            local_path=config['data']['local_raw_path']
        )
        print(f"Data extracted to: {raw_data_path}")
        
        # Step 2: Preprocess data
        print("Step 2: Preprocessing data...")
        processed_data_paths = preprocess_data(
            input_csv=raw_data_path,
            output_dir=config['data']['processed_dir'],
            test_size=config['data']['test_size'],
            random_state=config['seed']
        )
        print(f"Data preprocessed and saved to: {config['data']['processed_dir']}")
        
        # Step 3: Train model
        print("Step 3: Training model...")
        model_uri = train_model(
            train_data_paths={
                'X_train': processed_data_paths['X_train'],
                'y_train': processed_data_paths['y_train']
            },
            model_output_path=config['model']['output_dir'],
            n_estimators=config['model']['n_estimators'],
            max_depth=config['model']['max_depth'],
            random_state=config['seed']
        )
        print(f"Model trained and saved. URI: {model_uri}")
        
        # Step 4: Evaluate model
        print("Step 4: Evaluating model...")
        metrics = evaluate_model(
            model_uri=model_uri,
            test_data_paths={
                'X_test': processed_data_paths['X_test'],
                'y_test': processed_data_paths['y_test']
            },
            metrics_output_path=config['evaluation']['output_path']
        )
        print(f"Model evaluated. Metrics: {metrics}")
        print(f"Metrics saved to: {config['evaluation']['output_path']}")
        
        print("\nPipeline completed successfully!")
        print(f"View results in MLflow UI: mlflow ui")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Run MLflow pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == '__main__':
    main()

