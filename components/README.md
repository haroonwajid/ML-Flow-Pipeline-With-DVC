# Pipeline Components

This directory contains YAML component definitions for the MLflow-based MLOps pipeline.

## Component Files

1. **extract_data_component.yaml**
   - Extracts data using DVC pull or local file system
   - Inputs: `dvc_path`, `local_path`
   - Outputs: `raw_data_path`

2. **preprocess_data_component.yaml**
   - Preprocesses data and creates train/test split with feature scaling
   - Inputs: `input_csv`, `output_dir`, `test_size`, `random_state`
   - Outputs: `X_train`, `y_train`, `X_test`, `y_test`, `scaler`

3. **train_model_component.yaml**
   - Trains a RandomForestRegressor model
   - Inputs: `X_train_path`, `y_train_path`, `model_output_path`, `n_estimators`, `max_depth`, `random_state`
   - Outputs: `model_uri`, `model_path`

4. **evaluate_model_component.yaml**
   - Evaluates a trained model and computes regression metrics
   - Inputs: `model_uri`, `X_test_path`, `y_test_path`, `metrics_output_path`
   - Outputs: `mse`, `rmse`, `mae`, `r2`, `metrics_file`

## Component Structure

Each component YAML file follows this structure:

```yaml
name: Component Name
description: Component description
inputs:
  - name: input_name
    type: data_type
    description: Input description
    default: default_value
outputs:
  - name: output_name
    type: data_type
    description: Output description
implementation:
  type: python_function
  module: src.pipeline_components
  function: function_name
  parameters:
    param_name: {input: input_name}
  returns:
    output_name: {output: output_name}
```

## Usage

These components are used by the main pipeline orchestration script in `pipelines/pipeline.py`. The pipeline reads the configuration from `configs/config.yaml` and executes the components in sequence:

1. Extract Data → 2. Preprocess Data → 3. Train Model → 4. Evaluate Model

## MLflow Integration

All components integrate with MLflow for:
- Experiment tracking
- Parameter logging
- Metric logging
- Artifact storage
- Model versioning

## Notes

- These components use MLflow instead of Kubeflow Pipelines
- Components are implemented as Python functions in `src/pipeline_components.py`
- The YAML files define the component interface and metadata
- Actual execution is handled by the MLflow pipeline orchestration

