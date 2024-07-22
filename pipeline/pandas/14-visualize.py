#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df.drop(columns=['Weighted_Price'], inplace=True)
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

df.set_index('Date', inplace=True)

df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

df = df[df.index >= '2017-01-01']

df_daily = ({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})


df = df.resample('D').agg(df_daily)


plt.figure(figsize=(20, 10))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['High'], label='High')
plt.plot(df.index, df['Low'], label='Low')
plt.plot(df.index, df['Open'], label='Open')
plt.plot(df.index, df['Volume_(BTC)'], label='Volume_(BTC)')
plt.plot(df.index, df['Volume_(Currency)'], label='Volume_(Currency)')
plt.title('Daily Price Data from 2017 and Beyond')
plt.xlabel('Date')
plt.legend('Price')
plt.show()


