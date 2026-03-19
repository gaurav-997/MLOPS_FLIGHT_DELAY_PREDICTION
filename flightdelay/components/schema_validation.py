"""
Simple schema validation for flight delay prediction pipeline.
Validates data structure before ingestion to prevent downstream failures.
"""

import pandas as pd
from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
import sys


def validate_flights_schema(df: pd.DataFrame) -> bool:
    """
    Validate flights.csv schema.
    
    Args:
        df: Flights DataFrame
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If schema is invalid
    """
    try:
        logger.info("Validating flights schema...")
        
        required_columns = [
            "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
            "AIRLINE", "FLIGHT_NUMBER",
            "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY",
            "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY",
            "DISTANCE"
        ]
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Basic type checks
        if not pd.api.types.is_numeric_dtype(df['ARRIVAL_DELAY']):
            raise TypeError("ARRIVAL_DELAY must be numeric")
        
        if not pd.api.types.is_numeric_dtype(df['DISTANCE']):
            raise TypeError("DISTANCE must be numeric")
        
        logger.info("[PASS] Flights schema validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Flights schema validation failed: {str(e)}")
        raise CustomException(e, sys)


def validate_airports_schema(df: pd.DataFrame) -> bool:
    """
    Validate airports.csv schema.
    
    Args:
        df: Airports DataFrame
        
    Returns:
        True if valid
    """
    try:
        logger.info("Validating airports schema...")
        
        required_columns = [
            "IATA_CODE", "AIRPORT", "CITY", 
            "STATE", "COUNTRY", 
            "LATITUDE", "LONGITUDE"
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check coordinates are numeric
        if not pd.api.types.is_numeric_dtype(df['LATITUDE']):
            raise TypeError("LATITUDE must be numeric")
        
        if not pd.api.types.is_numeric_dtype(df['LONGITUDE']):
            raise TypeError("LONGITUDE must be numeric")
        
        logger.info("[PASS] Airports schema validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Airports schema validation failed: {str(e)}")
        raise CustomException(e, sys)


def validate_airlines_schema(df: pd.DataFrame) -> bool:
    """
    Validate airlines.csv schema.
    
    Args:
        df: Airlines DataFrame
        
    Returns:
        True if valid
    """
    try:
        logger.info("Validating airlines schema...")
        
        required_columns = ["IATA_CODE", "AIRLINE"]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("[PASS] Airlines schema validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Airlines schema validation failed: {str(e)}")
        raise CustomException(e, sys)


def validate_holidays_schema(df: pd.DataFrame) -> bool:
    """
    Validate holidays data schema.
    
    Args:
        df: Holidays DataFrame
        
    Returns:
        True if valid
    """
    try:
        logger.info("Validating holidays schema...")
        
        required_columns = ["date", "holiday_name", "is_holiday"]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("[PASS] Holidays schema validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Holidays schema validation failed: {str(e)}")
        raise CustomException(e, sys)


def validate_weather_schema(df: pd.DataFrame) -> bool:
    """
    Validate weather data schema.
    
    Args:
        df: Weather DataFrame
        
    Returns:
        True if valid
    """
    try:
        logger.info("Validating weather schema...")
        
        required_columns = ["date","AWND", "TMAX", "TMIN", "PRCP"] 
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("[PASS] Weather schema validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Weather schema validation failed: {str(e)}")
        raise CustomException(e, sys)


def validate_all_schemas(flights_df, airports_df, airlines_df, holidays_df, weather_df):
    """
    Validate all data sources at once.
    
    Args:
        flights_df: Flights DataFrame
        airports_df: Airports DataFrame
        airlines_df: Airlines DataFrame
        holidays_df: Holidays DataFrame
        weather_df: Weather DataFrame
        
    Returns:
        True if all validations pass
    """
    try:
        logger.info("Starting multi-source schema validation...")
        print("  [1/5] Validating flights schema...")
        validate_flights_schema(flights_df)
        print("Flights validated")
        logger.info("*" * 50)
        
        print("  [2/5] Validating airports schema...")
        validate_airports_schema(airports_df)
        print("Airports validated")
        logger.info("*" * 50)
        
        print("  [3/5] Validating airlines schema...")
        validate_airlines_schema(airlines_df)
        print("Airlines validated")
        logger.info("*" * 50)
        
        print("  [4/5] Validating holidays schema...")
        validate_holidays_schema(holidays_df)
        print("Holidays validated")
        logger.info("*" * 50)
        
        print("  [5/5] Validating weather schema...")
        validate_weather_schema(weather_df)
        print("Weather validated")
        logger.info("*" * 50)
        
        logger.info("[SUCCESS] All schema validations passed!")
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Test validation
    print("Schema Validation Module")
    print("Import and use in your data pipeline")
