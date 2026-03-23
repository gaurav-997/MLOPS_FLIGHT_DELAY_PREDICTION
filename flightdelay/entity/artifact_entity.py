from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    ingested_flights_path: str
    ingested_airports_path: str
    ingested_airlines_path: str
    ingested_holidays_path: str
    ingested_weather_path: str
@dataclass
class DataValidationArtifact:
    validated_flights_dir: str
    data_validation_status: bool
    validated_airports_dir: str
    validated_airlines_dir: str
    validated_holidays_dir: str
    validated_weather_dir: str
    data_drift_report_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    joined_data_file_path: str

@dataclass
class RegressionMetricArtifact:
    r2_score: float
    mae: float
    rmse: float
    mse: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: RegressionMetricArtifact
    test_metric_artifact: RegressionMetricArtifact
    best_model_name: str
    best_model_score: float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_score: float
    best_model_path: str
    trained_model_path: str
    train_metric: RegressionMetricArtifact
    best_model_metric: RegressionMetricArtifact
    report_file_path: str