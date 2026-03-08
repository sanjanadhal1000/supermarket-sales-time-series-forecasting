import pandas as pd                          # Data manipulation and analysis

def load_and_preprocess(path):               # Define a function that loads the dataset and performs preprocessing
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column into datetime format for time-series processing
    df.set_index('Date', inplace=True)       # Set the 'Date' column as the DataFrame index to enable time-based operations
    df = df.sort_index()                     # Sort the data by date to ensure chronological order

    daily = df.resample('D').sum()           # Resample the dataset to daily frequency and aggregate values using sum

    daily['lag_1'] = daily['Total'].shift(1) # Create a lag feature representing the sales value from 1 day earlier
    daily['lag_7'] = daily['Total'].shift(7) # Create another lag feature representing the sales value from 7 days earlier
    daily.dropna(inplace=True)               # Remove rows with missing values caused by the lag feature creation

    return daily                             # Return the processed DataFrame with lag features