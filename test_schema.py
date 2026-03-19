"""
Test script for simple schema validation.
Run this to verify your data schemas before pipeline execution.
"""

import pandas as pd
import sys
import os
from flightdelay.exception.exception import CustomException
from flightdelay.logging.logger import logger

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flightdelay.components.schema_validation import validate_all_schemas


def main():
    print("SCHEMA VALIDATION TEST")
    try:
        # Load all data
        print("\nLoading data sources...")
        df_flights = pd.read_csv("delay_data/flights_sample.csv")
        df_airports = pd.read_csv("delay_data/airports.csv")
        df_airlines = pd.read_csv("delay_data/airlines.csv")
        df_holidays = pd.read_csv("delay_data/holidays.csv")
        df_weather = pd.read_csv("delay_data/weather_data.csv")
        
        print(f"Loaded: {len(df_flights)} flights, {len(df_airports)} airports, {len(df_airlines)} airlines, {len(df_holidays)} holidays, {len(df_weather)} weather records")
        
        # Validate all schemas at once
        print("\nValidating all schemas...")
        validate_all_schemas(df_flights, df_airports, df_airlines, df_holidays, df_weather)
    
        print("ALL VALIDATIONS PASSED")
        
    except Exception as e:
        print("VALIDATION FAILED")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
