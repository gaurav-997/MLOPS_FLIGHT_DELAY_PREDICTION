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




