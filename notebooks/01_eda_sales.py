import pandas as pd              # loading, cleaning, grouping, resampling, data manipulation
import matplotlib.pyplot as plt  # plotting graphs

# Load Dataset
df = pd.read_csv("../data/raw/supermarket_sales.csv")
print(df.head())                 # First 5 rows, used for debugging

print(df.info())
print(df.isna().sum())           # Checks missing values

# Date Handling
df['Date'] = pd.to_datetime(df['Date']) # Converts 'Date' column to datetime format
df = df.sort_values('Date')             # Ensures chronological order

df.set_index('Date', inplace=True)      # Makes 'Date' the index

# Create Time-Series Target
daily_sales = df.resample('D').sum()    # Aggregates transactions into daily totals
print(daily_sales.head())

daily_sales.to_csv("../data/processed/daily_sales.csv") # Saves processed dataset

# Visual EDA
plt.figure(figsize=(12,5))
plt.plot(daily_sales.index, daily_sales['Total'])
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.savefig('../docs/screenshots/daily_sales_over_time.png')

# Add Rolling Mean
daily_sales['rolling_7'] = daily_sales['Total'].rolling(7).mean() # Smooths noise, shows trend

plt.figure(figsize=(12,5))
plt.plot(daily_sales['Total'], label='Actual')
plt.plot(daily_sales['rolling_7'], label='7-day MA')
plt.legend()
plt.savefig('../docs/screenshots/rolling_mean.png')

# Stationary Check (ADF Test)
from statsmodels.tsa.stattools import adfuller # Imports stationary test

result = adfuller(daily_sales['Total'])        # Tests if series is stationary
print(f"ADF Statistic: {result[0]:.2f}")       # ADF - Augmented Dickey-Fuller Test 
print(f"p-value: {result[1]:.2f}")

# Feature Engineering, Seasonality Signals
daily_sales['day'] = daily_sales.index.day
daily_sales['month'] = daily_sales.index.month
daily_sales['weekday'] = daily_sales.index.weekday

# Adds previous day and weekly memory, important for forecasting
daily_sales['lag_1'] = daily_sales['Total'].shift(1)
daily_sales['lag_7'] = daily_sales['Total'].shift(7)

daily_sales.dropna(inplace=True)