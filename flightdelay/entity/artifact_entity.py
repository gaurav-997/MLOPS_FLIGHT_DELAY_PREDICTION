from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    ingested_flights_path: str
    ingested_airports_path: str
    ingested_airlines_path: str
    ingested_holidays_path: str
    ingested_weather_path: str