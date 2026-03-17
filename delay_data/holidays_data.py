import pandas as pd
import holidays

def get_holidays(years=[2015]):
    us_holidays = holidays.US(years=years)

    data = []
    for date, name in us_holidays.items():
        data.append({
            "date": pd.to_datetime(date),
            "holiday_name": name,
            "is_holiday": 1
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = get_holidays([2023])
    print(df.head())