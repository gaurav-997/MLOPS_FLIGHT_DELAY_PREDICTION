import requests
import pandas as pd
from dotenv import load_dotenv
import os

# get tocken form here https://www.ncdc.noaa.gov/cdo-web/token

# Load API key
load_dotenv()
API_KEY = os.getenv("NOAA_API_KEY")


def get_weather(station="GHCND:USW00094728", start="2015-01-01", end="2015-01-12"):
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

    headers = {
        "token": API_KEY
    }

    params = {
        "datasetid": "GHCND",
        "stationid": station,
        "startdate": start,
        "enddate": end,
        "units": "metric",
        "limit": 1000
    }

    response = requests.get(url, headers=headers, params=params)

    data = response.json()

    df = pd.DataFrame(data.get("results", []))
    return df


# 🔥 NEW: transform to ML-friendly format
def transform_weather(df):
    print("Transforming weather data...")

    # Convert date to YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Pivot table → wide format
    df_pivot = df.pivot_table(
        index='date',
        columns='datatype',
        values='value',
        aggfunc='mean'
    ).reset_index()

    return df_pivot


if __name__ == "__main__":
    df = get_weather()

    # ✅ Apply transformation
    df = transform_weather(df)

    # Save final dataset
    df.to_csv("weather_data.csv", index=False)

    print("Saved weather_data.csv")
    print(df.head())