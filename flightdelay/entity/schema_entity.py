"""
Schema definitions for all data sources in the flight delay prediction project.
Uses Pydantic for runtime type validation and data quality checks.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class FlightSchema(BaseModel):
    """Schema for flights.csv - Main dataset"""
    YEAR: int = Field(..., ge=2000, le=2030, description="Flight year")
    MONTH: int = Field(..., ge=1, le=12, description="Flight month")
    DAY: int = Field(..., ge=1, le=31, description="Flight day")
    DAY_OF_WEEK: int = Field(..., ge=1, le=7, description="Day of week (1=Monday)")
    AIRLINE: str = Field(..., min_length=2, max_length=3, description="Airline IATA code")
    FLIGHT_NUMBER: int = Field(..., gt=0, description="Flight number")
    TAIL_NUMBER: str = Field(..., description="Aircraft tail number")
    ORIGIN_AIRPORT: str = Field(..., min_length=3, max_length=3, description="Origin airport IATA code")
    DESTINATION_AIRPORT: str = Field(..., min_length=3, max_length=3, description="Destination airport IATA code")
    SCHEDULED_DEPARTURE: int = Field(..., ge=0, le=2359, description="Scheduled departure time (HHMM)")
    DEPARTURE_TIME: Optional[int] = Field(None, ge=0, le=2359, description="Actual departure time")
    DEPARTURE_DELAY: Optional[float] = Field(None, description="Departure delay in minutes")
    TAXI_OUT: Optional[int] = Field(None, ge=0, description="Taxi out time in minutes")
    WHEELS_OFF: Optional[int] = Field(None, ge=0, le=2359, description="Wheels off time")
    SCHEDULED_TIME: int = Field(..., gt=0, description="Scheduled flight time in minutes")
    ELAPSED_TIME: Optional[int] = Field(None, gt=0, description="Actual elapsed time")
    AIR_TIME: Optional[int] = Field(None, gt=0, description="Time in air")
    DISTANCE: int = Field(..., gt=0, description="Flight distance in miles")
    WHEELS_ON: Optional[int] = Field(None, ge=0, le=2359, description="Wheels on time")
    TAXI_IN: Optional[int] = Field(None, ge=0, description="Taxi in time")
    SCHEDULED_ARRIVAL: int = Field(..., ge=0, le=2359, description="Scheduled arrival time")
    ARRIVAL_TIME: Optional[int] = Field(None, ge=0, le=2359, description="Actual arrival time")
    ARRIVAL_DELAY: Optional[float] = Field(None, description="Arrival delay in minutes (TARGET)")
    DIVERTED: int = Field(..., ge=0, le=1, description="Flight diverted flag")
    CANCELLED: int = Field(..., ge=0, le=1, description="Flight cancelled flag")
    CANCELLATION_REASON: Optional[str] = Field(None, max_length=1, description="Cancellation reason code")
    AIR_SYSTEM_DELAY: Optional[float] = Field(None, ge=0, description="Air system delay minutes")
    SECURITY_DELAY: Optional[float] = Field(None, ge=0, description="Security delay minutes")
    AIRLINE_DELAY: Optional[float] = Field(None, ge=0, description="Airline delay minutes")
    LATE_AIRCRAFT_DELAY: Optional[float] = Field(None, ge=0, description="Late aircraft delay minutes")
    WEATHER_DELAY: Optional[float] = Field(None, ge=0, description="Weather delay minutes")

    @field_validator('ORIGIN_AIRPORT', 'DESTINATION_AIRPORT')
    @classmethod
    def validate_iata_code(cls, v: str) -> str:
        """Ensure IATA codes are uppercase"""
        return v.upper()

    @field_validator('AIRLINE')
    @classmethod
    def validate_airline_code(cls, v: str) -> str:
        """Ensure airline codes are uppercase"""
        return v.upper()

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class AirportSchema(BaseModel):
    """Schema for airports.csv - Static metadata"""
    IATA_CODE: str = Field(..., min_length=3, max_length=3, description="Airport IATA code")
    AIRPORT: str = Field(..., min_length=1, description="Airport name")
    CITY: str = Field(..., min_length=1, description="City name")
    STATE: str = Field(..., min_length=2, max_length=2, description="State code")
    COUNTRY: str = Field(..., min_length=2, max_length=3, description="Country code")
    LATITUDE: float = Field(..., ge=-90, le=90, description="Airport latitude")
    LONGITUDE: float = Field(..., ge=-180, le=180, description="Airport longitude")

    @field_validator('IATA_CODE')
    @classmethod
    def validate_iata_code(cls, v: str) -> str:
        """Ensure IATA code is uppercase"""
        return v.upper()

    @field_validator('STATE')
    @classmethod
    def validate_state_code(cls, v: str) -> str:
        """Ensure state code is uppercase"""
        return v.upper()

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class AirlineSchema(BaseModel):
    """Schema for airlines.csv - Static metadata"""
    IATA_CODE: str = Field(..., min_length=2, max_length=3, description="Airline IATA code")
    AIRLINE: str = Field(..., min_length=1, description="Airline name")

    @field_validator('IATA_CODE')
    @classmethod
    def validate_iata_code(cls, v: str) -> str:
        """Ensure IATA code is uppercase"""
        return v.upper()

    class Config:
        str_strip_whitespace = True
        validate_assignment = True


class HolidaySchema(BaseModel):
    """Schema for holidays data - Generated from holidays library"""
    date: datetime = Field(..., description="Holiday date")
    holiday_name: str = Field(..., min_length=1, description="Name of the holiday")
    is_holiday: int = Field(..., ge=0, le=1, description="Holiday flag (1=holiday, 0=not)")

    class Config:
        validate_assignment = True


class WeatherSchema(BaseModel):
    """Schema for weather data from NOAA API"""
    date: str = Field(..., description="Date of weather observation (ISO format)")
    station: str = Field(..., description="Weather station ID")
    datatype: str = Field(..., description="Type of measurement (TMAX, TMIN, PRCP, etc.)")
    value: float = Field(..., description="Measurement value")
    attributes: Optional[str] = Field(None, description="Measurement attributes/flags")

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Ensure date is in ISO format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD)")

    class Config:
        validate_assignment = True


# Schema mapping for easy access
SCHEMA_MAPPING = {
    "flights": FlightSchema,
    "airports": AirportSchema,
    "airlines": AirlineSchema,
    "holidays": HolidaySchema,
    "weather": WeatherSchema
}
