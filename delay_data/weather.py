import requests
import pandas as pd
#  get tocken form here https://www.ncdc.noaa.gov/cdo-web/token

API_KEY = "YOUR_API_KEY"

def get_weather(station="GHCND:USW00094728", start="2015-01-01", end="2015-01-05"):
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
    print(df.head())