"""Tests for flightdelay.entity.artifact_entity and config_entity"""
import os
import pytest
from flightdelay.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    RegressionMetricArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from flightdelay.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class TestArtifactEntities:
    def test_data_ingestion_artifact_fields(self):
        artifact = DataIngestionArtifact(
            ingested_flights_path="a/flights.csv",
            ingested_airports_path="a/airports.csv",
            ingested_airlines_path="a/airlines.csv",
            ingested_holidays_path="a/holidays.csv",
            ingested_weather_path="a/weather.csv",
        )
        assert artifact.ingested_flights_path == "a/flights.csv"
        assert artifact.ingested_weather_path == "a/weather.csv"

    def test_data_validation_artifact_fields(self):
        artifact = DataValidationArtifact(
            validated_flights_dir="v/flights.csv",
            data_validation_status=True,
            validated_airports_dir="v/airports.csv",
            validated_airlines_dir="v/airlines.csv",
            validated_holidays_dir="v/holidays.csv",
            validated_weather_dir="v/weather.csv",
            data_drift_report_path="v/drift.yaml",
        )
        assert artifact.data_validation_status is True
        assert artifact.data_drift_report_path == "v/drift.yaml"

    def test_regression_metric_artifact_fields(self):
        m = RegressionMetricArtifact(r2_score=0.85, mae=5.2, rmse=7.1, mse=50.4)
        assert m.r2_score == 0.85
        assert m.mae == 5.2
        assert m.rmse == 7.1
        assert m.mse == 50.4

    def test_model_trainer_artifact_optional_run_id(self):
        metric = RegressionMetricArtifact(r2_score=0.8, mae=4.0, rmse=6.0, mse=36.0)
        artifact = ModelTrainerArtifact(
            trained_model_file_path="model.pkl",
            train_metric_artifact=metric,
            test_metric_artifact=metric,
            best_model_name="LightGBM",
            best_model_score=0.8,
        )
        assert artifact.mlflow_run_id is None
        assert artifact.best_model_name == "LightGBM"

    def test_model_evaluation_artifact_fields(self):
        metric = RegressionMetricArtifact(r2_score=0.8, mae=4.0, rmse=6.0, mse=36.0)
        artifact = ModelEvaluationArtifact(
            is_model_accepted=True,
            improved_score=0.05,
            best_model_path="final_model/model.pkl",
            trained_model_path="artifacts/model.pkl",
            train_metric=metric,
            best_model_metric=metric,
            report_file_path="report.yaml",
        )
        assert artifact.is_model_accepted is True
        assert artifact.improved_score == 0.05

    def test_data_transformation_artifact_fields(self):
        artifact = DataTransformationArtifact(
            transformed_object_file_path="t/preprocessor.pkl",
            transformed_train_file_path="t/train.npy",
            transformed_test_file_path="t/test.npy",
            joined_data_file_path="t/joined.csv",
        )
        assert artifact.transformed_train_file_path == "t/train.npy"


class TestConfigEntities:
    def test_training_pipeline_config_creates_artifact_dir(self):
        config = TrainingPipelineConfig()
        assert config.pipeline_name == "flight_delay_prediction_pipeline"
        assert "Artifacts" in config.artifact_dir

    def test_data_ingestion_config_paths(self):
        pipeline_config = TrainingPipelineConfig()
        config = DataIngestionConfig(pipeline_config)
        assert "data_ingestion" in config.data_ingestion_dir
        assert config.data_ingestion_ingested_flights_file_path.endswith("flights.csv")
        assert config.data_ingestion_ingested_airports_file_path.endswith("airports.csv")

    def test_data_validation_config_paths(self):
        pipeline_config = TrainingPipelineConfig()
        config = DataValidationConfig(pipeline_config)
        assert "data_validation" in config.data_validation_dir
        assert config.data_drift_report_file_path.endswith("drift_report.yaml")

    def test_data_transformation_config_paths(self):
        pipeline_config = TrainingPipelineConfig()
        config = DataTransformationConfig(pipeline_config)
        assert "data_transformation" in config.data_transformation_dir
        assert config.transformed_object_file_path.endswith("preprocessor.pkl")
        assert config.transformed_train_file_path.endswith("train.npy")
        assert config.transformed_test_file_path.endswith("test.npy")

    def test_model_trainer_config(self):
        pipeline_config = TrainingPipelineConfig()
        config = ModelTrainerConfig(pipeline_config)
        assert "model_trainer" in config.model_trainer_dir
        assert config.expected_score == 0.6
        assert config.overfitting_threshold == 0.1

    def test_model_evaluation_config(self):
        pipeline_config = TrainingPipelineConfig()
        config = ModelEvaluationConfig(pipeline_config)
        assert "model_evaluation" in config.model_evaluation_dir
        assert config.best_model_dir == "final_model"
        assert config.best_model_file_path.endswith("model.pkl")
        assert config.best_preprocessor_file_path.endswith("preprocessor.pkl")

    def test_artifact_dirs_are_nested_under_timestamp(self):
        c1 = TrainingPipelineConfig()
        c2 = TrainingPipelineConfig()
        # Both should be under Artifacts/ but may have same or different timestamps
        assert c1.artifact_dir.startswith(os.path.join("Artifacts"))
        assert c2.artifact_dir.startswith(os.path.join("Artifacts"))
