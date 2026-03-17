import requests
import pandas as pd
from dotenv import load_dotenv
import os
#  get tocken form here https://www.ncdc.noaa.gov/cdo-web/token

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


if __name__ == "__main__":
    df = get_weather()
    df.to_csv("weather_data.csv", index=False)
    print(df.head())