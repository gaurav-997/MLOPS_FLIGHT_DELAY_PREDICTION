import pandas as pd

df = pd.read_csv("flights.csv", nrows=100000)
df.to_csv("flights_sample.csv", index=False)