import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.pyfunc import PythonModel


DEFAULT_EXPERIMENT_NAME = "capstone-model-experiment"
DEFAULT_ARTIFACT_PATH = "my_model"
RUN_INFO_PATH = Path("models/latest_run_info.json")


class ConstantPredictionModel(PythonModel):
    def predict(self, context, model_input):
        _ = context
        return [1] * len(model_input)


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


def log_model_with_compatible_api(artifact_path: str):
    python_model = ConstantPredictionModel()
    try:
        return mlflow.pyfunc.log_model(
            name=artifact_path,
            python_model=python_model,
        )
    except TypeError:
        return mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=python_model,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
    )
    parser.add_argument(
        "--artifact-path",
        default=os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
    )
    args = parser.parse_args()

    validate_artifact_path(args.artifact_path)
    tracking_uri = configure_tracking()
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("artifact_path", args.artifact_path)
        model_info = log_model_with_compatible_api(args.artifact_path)
        RUN_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUN_INFO_PATH.write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "artifact_path": args.artifact_path,
                    "experiment_name": args.experiment_name,
                    "tracking_uri": tracking_uri,
                    "model_uri": getattr(model_info, "model_uri", None),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "Model logged successfully:",
            f"run_id={run.info.run_id}",
            f"artifact_path={args.artifact_path}",
        )


if __name__ == "__main__":
    main()
