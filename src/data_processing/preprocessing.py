import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    daily = df.resample('D').sum()

    daily['lag_1'] = daily['Total'].shift(1)
    daily['lag_7'] = daily['Total'].shift(7)
    daily.dropna(inplace=True)

    return daily