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