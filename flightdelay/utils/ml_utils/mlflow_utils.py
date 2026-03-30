"""
MLflow Utility Functions
Handles experiment tracking, model registry, and DagsHub integration
for the Flight Delay Prediction MLOps pipeline.
"""

import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException


# ─────────────────────────────────────────────
# DagsHub / Tracking URI Configuration
# ─────────────────────────────────────────────

DAGSHUB_USER = os.getenv("DAGSHUB_USER", "chauhan7gaurav")
REPO_NAME = "MLOPS_FLIGHT_DELAY_PREDICTION"
EXPERIMENT_NAME = "FlightDelay_Prediction"
MODEL_NAME = "FlightDelayModel"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.mlflow"
)


def setup_mlflow() -> None:
    """
    Configure MLflow to point at the DagsHub tracking server.
    Call once at pipeline startup (or let each component call it).

    Environment variables (set before running):
        MLFLOW_TRACKING_URI   – overrides the default DagsHub URI
        MLFLOW_TRACKING_USERNAME – DagsHub username (for auth)
        MLFLOW_TRACKING_PASSWORD – DagsHub access-token / password
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
        logger.info(f"MLflow experiment   : {EXPERIMENT_NAME}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────

def log_model_training(
    best_model,
    best_model_name: str,
    train_r2: float,
    test_r2: float,
    test_mae: float,
    test_rmse: float,
    test_mse: float,
    expected_score: float,
    overfitting_threshold: float,
    features_count: int,
) -> str:
    """
    Log training params, metrics, and the sklearn model artifact to MLflow.

    Returns:
        run_id (str) – the MLflow run ID of the logged run.
    """
    try:
        setup_mlflow()

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started  : {run_id}")

            # ── Parameters ──────────────────────────────
            mlflow.log_param("model_name",             best_model_name)
            mlflow.log_param("expected_r2",            expected_score)
            mlflow.log_param("overfitting_threshold",  overfitting_threshold)
            mlflow.log_param("features_count",         features_count)

            # ── Metrics ─────────────────────────────────
            mlflow.log_metric("train_r2",  train_r2)
            mlflow.log_metric("test_r2",   test_r2)
            mlflow.log_metric("test_mae",  test_mae)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mse",  test_mse)

            # Derived / diagnostic metrics
            mlflow.log_metric("overfitting_gap", round(train_r2 - test_r2, 6))

            # ── Model artifact ───────────────────────────
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
            logger.info(f"Model logged & registered as '{MODEL_NAME}' (run={run_id})")

        return run_id

    except Exception as e:
        raise CustomException(e, sys)


def log_model_evaluation(
    run_id: str,
    is_model_accepted: bool,
    improvement: float,
    trained_r2: float,
    trained_mae: float,
    trained_rmse: float,
    production_r2: float = None,
    production_mae: float = None,
    production_rmse: float = None,
) -> None:
    """
    Append evaluation results to an existing MLflow run.
    If *run_id* is None a new run is created.
    """
    try:
        setup_mlflow()

        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_param("is_model_accepted", is_model_accepted)

            mlflow.log_metric("eval_improvement",   improvement)
            mlflow.log_metric("eval_trained_r2",    trained_r2)
            mlflow.log_metric("eval_trained_mae",   trained_mae)
            mlflow.log_metric("eval_trained_rmse",  trained_rmse)

            if production_r2 is not None:
                mlflow.log_metric("eval_production_r2",   production_r2)
                mlflow.log_metric("eval_production_mae",  production_mae)
                mlflow.log_metric("eval_production_rmse", production_rmse)

            logger.info(
                f"Evaluation results logged to run {run.info.run_id} "
                f"(accepted={is_model_accepted}, improvement={improvement:.4f})"
            )

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────────
# Model Registry helpers
# ─────────────────────────────────────────────

def register_model(run_id: str) -> int:
    """
    Register the model artifact from *run_id* in the MLflow Model Registry.

    Returns:
        version (int) – the newly created model version number.
    """
    try:
        setup_mlflow()
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)
        version = int(result.version)
        logger.info(f"Model registered: {MODEL_NAME} version {version}")
        return version
    except Exception as e:
        raise CustomException(e, sys)


def promote_model_to_production(version: int) -> None:
    """
    Transition a registered model version to the 'Production' stage and
    archive the previously active Production version (if any).

    Args:
        version (int): Model version to promote.
    """
    try:
        setup_mlflow()
        client = MlflowClient()

        # Archive any existing Production versions first
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if mv.current_stage == "Production":
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=int(mv.version),
                    stage="Archived",
                    archive_existing_versions=False,
                )
                logger.info(f"Archived previous production version: {mv.version}")

        # Promote new version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=False,
        )
        logger.info(f"Promoted {MODEL_NAME} v{version} → Production")

    except Exception as e:
        raise CustomException(e, sys)


def get_production_model_uri() -> str | None:
    """
    Return the model URI for the current Production version, or None if
    no Production version exists.
    """
    try:
        setup_mlflow()
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            logger.info(f"No Production version found for '{MODEL_NAME}'")
            return None
        uri = f"models:/{MODEL_NAME}/Production"
        logger.info(f"Production model URI: {uri}")
        return uri
    except Exception as e:
        raise CustomException(e, sys)
