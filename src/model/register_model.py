import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


DEFAULT_EXPERIMENT_NAME = "capstone-model-experiment"
DEFAULT_ARTIFACT_PATH = "my_model"
DEFAULT_REGISTERED_MODEL_NAME = "capstone-model"
RUN_ID_PATTERN = r"[0-9a-f]{32}"
MLFLOW_MODEL_FILE = "MLmodel"
RUN_INFO_PATH = Path("models/latest_run_info.json")


def configure_tracking() -> str:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file://{Path('mlruns').resolve()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def validate_artifact_path(artifact_path: str) -> None:
    if not artifact_path or not artifact_path.strip():
        raise ValueError("artifact_path must be a non-empty string")


def get_saved_run_info() -> Dict[str, str]:
    if not RUN_INFO_PATH.exists():
        return {}
    return json.loads(RUN_INFO_PATH.read_text(encoding="utf-8"))


def get_latest_run_id(experiment_name: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' does not exist. "
            "Run model_evaluation.py first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"No runs found in experiment '{experiment_name}'. "
            "Run model_evaluation.py first."
        )
    run_id = runs.iloc[0]["run_id"]
    validate_run_id(run_id)
    return run_id


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(RUN_ID_PATTERN, run_id):
        raise ValueError("run_id must be a 32-character lowercase hex string")


def model_artifact_exists(
    client: MlflowClient, run_id: str, artifact_path: str
) -> bool:
    root_artifacts = client.list_artifacts(run_id)
    for artifact in root_artifacts:
        if artifact.path == artifact_path and artifact.is_dir:
            children = client.list_artifacts(run_id, artifact_path)
            child_paths = {child.path for child in children}
            if f"{artifact_path}/{MLFLOW_MODEL_FILE}" in child_paths:
                return True
    return False


def get_logged_model_uri(
    client: MlflowClient,
    experiment_name: str,
    run_id: str,
    artifact_path: str,
) -> Optional[str]:
    """Resolve a logged-model URI for the given run."""
    validate_run_id(run_id)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    try:
        logged_models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
        )
    except MlflowException as exc:
        print(
            f"Warning: failed to search logged models for run '{run_id}': "
            f"{exc}",
            file=sys.stderr,
        )
        return None

    fallback_model_uri = None
    for model in logged_models:
        model_name = getattr(model, "name", "")
        model_uri = getattr(model, "model_uri", None)
        if model_name == artifact_path and model_uri:
            return model_uri
        if fallback_model_uri is None and model_uri:
            fallback_model_uri = model_uri
    return fallback_model_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registered-model-name",
        default=os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME",
            DEFAULT_REGISTERED_MODEL_NAME,
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
    )
    parser.add_argument(
        "--artifact-path",
        default=os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
    )
    parser.add_argument("--run-id", default=os.getenv("MLFLOW_RUN_ID"))
    args = parser.parse_args()

    validate_artifact_path(args.artifact_path)
    saved_run_info = get_saved_run_info()
    tracking_uri = configure_tracking()
    mlflow.set_experiment(args.experiment_name)

    run_id = args.run_id or saved_run_info.get("run_id") or get_latest_run_id(
        args.experiment_name
    )
    validate_run_id(run_id)
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{args.artifact_path}"
    # Priority:
    # run artifact URI -> MLflow 3 logged model URI -> saved model URI.
    if not model_artifact_exists(client, run_id, args.artifact_path):
        model_uri = get_logged_model_uri(
            client,
            args.experiment_name,
            run_id,
            args.artifact_path,
        ) or (
            saved_run_info.get("model_uri")
            if saved_run_info.get("run_id") == run_id
            else None
        )

    if not model_uri:
        root_artifacts = [a.path for a in client.list_artifacts(run_id)]
        raise FileNotFoundError(
            "Unable to find a logged model with artifact_path "
            f"'{args.artifact_path}' in run '{run_id}'. "
            f"Available root artifacts: {root_artifacts}"
        )
    try:
        result = mlflow.register_model(
            model_uri=model_uri,
            name=args.registered_model_name,
        )
    except MlflowException as exc:
        raise RuntimeError(
            f"Model registration failed for uri '{model_uri}' "
            f"on tracking uri '{tracking_uri}'."
        ) from exc

    print(
        "Model registered successfully:",
        f"name={result.name}",
        f"version={result.version}",
        f"model_uri={model_uri}",
    )


if __name__ == "__main__":
    main()

