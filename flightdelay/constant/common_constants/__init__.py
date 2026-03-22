# common constants for the project
PIPELINE_NAME = 'flight_delay_prediction_pipeline'
TARGET_COLUMN = 'ARRIVAL_DELAY'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
FINAL_DATA_FILE_NAME = 'final_delay_data.csv'
ARTIFACT_DIR_NAME = 'Artifacts'

# schema validation constants
SCHEMA_VALIDATION_DIR_NAME = 'schema_validation'


# data ingestion constants 
DATA_INGESTION_DIR_NAME = 'data_ingestion'
DATA_INGESTION_INGESTED_DIR = 'ingested'
DATA_INGESTION_INGESTED_FLIGHTS_FILE_NAME = 'flights.csv'
DATA_INGESTION_INGESTED_AIRPORTS_FILE_NAME = 'airports.csv'
DATA_INGESTION_INGESTED_AIRLINES_FILE_NAME = 'airlines.csv'
DATA_INGESTION_INGESTED_HOLIDAYS_FILE_NAME = 'holidays.csv'
DATA_INGESTION_INGESTED_WEATHER_FILE_NAME = 'weather.csv'

# data validation constants
DATA_VALIDATION_DIR_NAME = 'data_validation'
DATA_VALIDATION_VALID_DIR = 'validated'
DATA_VALIDATION_INVALID_DIR = 'invalidated'
DATA_DRIFT_REPORT_FILE_NAME = 'drift_report.yaml'

# data transformation constants
DATA_TRANSFORMATION_DIR_NAME = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = 'transformed_object'
PREPROCESSING_OBJECT_FILE_NAME = 'preprocessor.pkl'
FINAL_JOINED_DATA_FILE_NAME = 'flights_joined.csv'
TRANSFORMED_TRAIN_FILE_NAME = 'train.npy'
TRANSFORMED_TEST_FILE_NAME = 'test.npy'

# model training constants
MODEL_TRAINER_DIR_NAME = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME = 'model.pkl'
MODEL_TRAINER_EXPECTED_SCORE = 0.6  # Min R² for regression
MODEL_TRAINER_OVERFITTING_THRESHOLD = 0.1  # Max train-test gap

