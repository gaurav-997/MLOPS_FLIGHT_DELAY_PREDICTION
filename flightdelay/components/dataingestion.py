import os
import sys

import pandas as pd
from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
from flightdelay.entity.config_entity import DataIngestionConfig
from flightdelay.entity.artifact_entity import DataIngestionArtifact
from flightdelay.components.schema_validation import validate_all_schemas


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            # Source data paths (raw data location)
            self.source_flights_path = "delay_data/flights_sample.csv"
            self.source_airports_path = "delay_data/airports.csv"
            self.source_airlines_path = "delay_data/airlines.csv"
            self.source_holidays_path = "delay_data/holidays.csv"
            self.source_weather_path = "delay_data/weather_data.csv"
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_to_ingested_dir(self, dataframe: pd.DataFrame, destination_path: str):
        try:
            dir_path = os.path.dirname(destination_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(destination_path, index=False)
            logger.info(f"Exported data to: {destination_path}")
            return destination_path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion process...")

            # Step 1: Load all raw data from source
            print("\n[Step 1/3] Loading raw data from sources...")
            flights_df = self.load_data(self.source_flights_path)
            print(f"Loaded flights: {len(flights_df)} rows")

            airports_df = self.load_data(self.source_airports_path)
            print(f" Loaded airports: {len(airports_df)} rows")

            airlines_df = self.load_data(self.source_airlines_path)
            print(f" Loaded airlines: {len(airlines_df)} rows")

            holidays_df = self.load_data(self.source_holidays_path)
            print(f" Loaded holidays: {len(holidays_df)} rows")

            weather_df = self.load_data(self.source_weather_path)
            print(f" Loaded weather: {len(weather_df)} rows")

            # Step 2: Validate schemas
            print("\n[Step 2/3] Validating schemas...")
            validate_all_schemas(flights_df, airports_df, airlines_df, holidays_df, weather_df)
            print(" All schemas validated successfully")

            # Step 3: Export to ingested directory
            print("\n[Step 3/3] Exporting to ingested directory...")

            self.export_data_to_ingested_dir(
                flights_df,
                self.data_ingestion_config.data_ingestion_ingested_flights_file_path
            )
            print(f" Flights saved to: {self.data_ingestion_config.data_ingestion_ingested_flights_file_path}")

            self.export_data_to_ingested_dir(
                airports_df,
                self.data_ingestion_config.data_ingestion_ingested_airports_file_path
            )
            print(f" Airports saved to: {self.data_ingestion_config.data_ingestion_ingested_airports_file_path}")

            self.export_data_to_ingested_dir(
                airlines_df,
                self.data_ingestion_config.data_ingestion_ingested_airlines_file_path
            )
            print(f" Airlines saved to: {self.data_ingestion_config.data_ingestion_ingested_airlines_file_path}")

            self.export_data_to_ingested_dir(
                holidays_df,
                self.data_ingestion_config.data_ingestion_ingested_holidays_file_path
            )
            print(f" Holidays saved to: {self.data_ingestion_config.data_ingestion_ingested_holidays_file_path}")

            self.export_data_to_ingested_dir(
                weather_df,
                self.data_ingestion_config.data_ingestion_ingested_weather_file_path
            )
            print(f" Weather saved to: {self.data_ingestion_config.data_ingestion_ingested_weather_file_path}")

            # Create artifact
            data_ingestion_artifact = DataIngestionArtifact(
                ingested_flights_path=self.data_ingestion_config.data_ingestion_ingested_flights_file_path,
                ingested_airports_path=self.data_ingestion_config.data_ingestion_ingested_airports_file_path,
                ingested_airlines_path=self.data_ingestion_config.data_ingestion_ingested_airlines_file_path,
                ingested_holidays_path=self.data_ingestion_config.data_ingestion_ingested_holidays_file_path,
                ingested_weather_path=self.data_ingestion_config.data_ingestion_ingested_weather_file_path
            )
            logger.info("Data ingestion completed successfully")
            return data_ingestion_artifact

        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise CustomException(e, sys)
