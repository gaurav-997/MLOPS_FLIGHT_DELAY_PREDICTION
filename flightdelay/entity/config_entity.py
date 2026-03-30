import os 
import sys
from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
from flightdelay.constant import common_constants
from datetime import datetime


class TrainingPipelineConfig:
    def __init__(self):
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.pipeline_name = common_constants.PIPELINE_NAME
            self.artifact_dir_name = common_constants.ARTIFACT_DIR_NAME
            self.artifact_dir = os.path.join(self.artifact_dir_name, timestamp)
        except Exception as e:
            raise CustomException(e, sys)
        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, common_constants.DATA_INGESTION_DIR_NAME)
            self.data_ingestion_ingested_dir = os.path.join(self.data_ingestion_dir, common_constants.DATA_INGESTION_INGESTED_DIR)
            self.data_ingestion_ingested_flights_file_path = os.path.join(self.data_ingestion_ingested_dir, common_constants.DATA_INGESTION_INGESTED_FLIGHTS_FILE_NAME)
            self.data_ingestion_ingested_airports_file_path = os.path.join(self.data_ingestion_ingested_dir, common_constants.DATA_INGESTION_INGESTED_AIRPORTS_FILE_NAME)
            self.data_ingestion_ingested_airlines_file_path = os.path.join(self.data_ingestion_ingested_dir, common_constants.DATA_INGESTION_INGESTED_AIRLINES_FILE_NAME)
            self.data_ingestion_ingested_holidays_file_path = os.path.join(self.data_ingestion_ingested_dir, common_constants.DATA_INGESTION_INGESTED_HOLIDAYS_FILE_NAME)
            self.data_ingestion_ingested_weather_file_path = os.path.join(self.data_ingestion_ingested_dir, common_constants.DATA_INGESTION_INGESTED_WEATHER_FILE_NAME)

        except Exception as e:
            raise CustomException(e, sys)
        
class DataValidationConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, common_constants.DATA_VALIDATION_DIR_NAME)
            self.data_validation_valid_dir = os.path.join(self.data_validation_dir, common_constants.DATA_VALIDATION_VALID_DIR)
            self.data_validation_invalid_dir = os.path.join(self.data_validation_dir, common_constants.DATA_VALIDATION_INVALID_DIR)
            self.data_drift_report_file_path = os.path.join(self.data_validation_dir, common_constants.DATA_DRIFT_REPORT_FILE_NAME)
        except Exception as e:
            raise CustomException(e, sys)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, common_constants.DATA_TRANSFORMATION_DIR_NAME)
            self.transformed_data_dir = os.path.join(self.data_transformation_dir, common_constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
            self.transformed_object_dir = os.path.join(self.data_transformation_dir, common_constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)
            self.transformed_object_file_path = os.path.join(self.transformed_object_dir, common_constants.PREPROCESSING_OBJECT_FILE_NAME)
            self.final_joined_data_path = os.path.join(self.data_transformation_dir, common_constants.FINAL_JOINED_DATA_FILE_NAME)
            self.transformed_train_file_path = os.path.join(self.transformed_data_dir, common_constants.TRANSFORMED_TRAIN_FILE_NAME)
            self.transformed_test_file_path = os.path.join(self.transformed_data_dir, common_constants.TRANSFORMED_TEST_FILE_NAME)
        except Exception as e:
            raise CustomException(e, sys)


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, common_constants.MODEL_TRAINER_DIR_NAME)
            self.trained_model_dir = os.path.join(self.model_trainer_dir, common_constants.MODEL_TRAINER_TRAINED_MODEL_DIR)
            self.trained_model_file_path = os.path.join(self.trained_model_dir, common_constants.MODEL_TRAINER_TRAINED_MODEL_NAME)
            self.expected_score = common_constants.MODEL_TRAINER_EXPECTED_SCORE
            self.overfitting_threshold = common_constants.MODEL_TRAINER_OVERFITTING_THRESHOLD
        except Exception as e:
            raise CustomException(e, sys)


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.model_evaluation_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                common_constants.MODEL_EVALUATION_DIR_NAME
            )
            self.report_file_path = os.path.join(
                self.model_evaluation_dir,
                common_constants.MODEL_EVALUATION_REPORT_NAME
            )
            self.change_threshold = common_constants.MODEL_EVALUATION_CHANGED_THRESHOLD
            self.best_model_dir = common_constants.BEST_MODEL_DIR
            self.best_model_file_path = os.path.join(
                self.best_model_dir, common_constants.BEST_MODEL_FILE_NAME
            )
            self.best_preprocessor_file_path = os.path.join(
                self.best_model_dir, common_constants.BEST_PREPROCESSOR_FILE_NAME
            )
        except Exception as e:
            raise CustomException(e, sys)
