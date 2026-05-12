# register model

import json
import mlflow
import mlflow.sklearn
from src.logger import logging
import os
import dagshub
import pickle

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logging.error('Error loading model from %s: %s', file_path, e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def register_model(model_name: str, model_obj=None):
    """Register the model to the MLflow Model Registry using local backend."""
    try:
        # Use local MLflow backend for registration
        local_mlflow_uri = "file:///mlflow"
        mlflow.set_tracking_uri(local_mlflow_uri)
        
        logging.info('Using local MLflow backend for model registration')
        mlflow.set_experiment("local-model-registry")
        
        # Log model locally
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logging.info(f'Logging model to local run: {run_id}')
            mlflow.sklearn.log_model(model_obj, "model")
        
        # Now register from local run
        model_uri = f"runs:/{run_id}/model"
        logging.info(f'Registering model from local URI: {model_uri}')
        
        client = mlflow.tracking.MlflowClient()
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        
        logging.info(f'✅ Model {model_name} version {model_version.version} successfully registered and transitioned to Production!')
        logging.info(f'View your model at: file:///mlflow')
        
        return model_version
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        # Load the model from local file
        model_obj = load_model('./models/model.pkl')
        
        model_name = "my_model"
        register_model(model_name, model_obj)
        
        logging.info('\n🎉 Model registration completed successfully!')
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
