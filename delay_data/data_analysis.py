import pandas as pd

def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
        columns = data.columns
        print(f"Successfully read {file_path} with columns: {columns}")
        print(f"Shape of the data: {data.shape}")
        return data
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

df_flights = read_data("flights_sample.csv")
df_holidays = read_data("holidays.csv")
df_weather = read_data("weather_data.csv")
df_airports = read_data("airports.csv")
df_airlines = read_data("airlines.csv")

print("Flights data sample:")
print(df_flights.head())
print("\nMissing values in flights data:")
print(df_flights.isnull().sum())
print("\n how many delayed vs on-time flights:")
df_flights['delay'] = (df_flights['ARRIVAL_DELAY'] > 15).astype(int)
print(df_flights['delay'].value_counts())

print("\nHolidays data sample:")
print(df_holidays.head())
print("\nWeather data sample:")
print(df_weather.head())
print("\nAirports data sample:")
print(df_airports.head())
print("\nUnique origin airports in flights data:", df_flights['ORIGIN_AIRPORT'].nunique())
print("Unique IATA codes in airports data:", df_airports['IATA_CODE'].nunique())
print("\nAirlines data sample:")
print(df_airlines.head())
