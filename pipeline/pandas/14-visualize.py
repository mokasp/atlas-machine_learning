#!/usr/bin/env python3
""""""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


def transform_df(df):

    # drop weighted price column
    df = df.drop('Weighted_Price', axis=1)

    # rename the timestampe column to date and convert every column
    #entry to date format
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df['Date'] = df['Date'].dt.date

    # set date column to be the index
    df = df.set_index('Date')

    # fill empty/NaN entries in close column with the value in the previous
    # row
    df['Close'] = df['Close'].ffill()

    # fill empty/NaN values from high, low, and open columns with entry from
    # close column
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # fill empty/NaN values in volume btc and volume currency to 0
    df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

    # sort all rows by date
    df.sort_index()

    # aggregate data at daily intervals
    df = df.groupby('Date').agg({'High': 'max', 'Low': 'min', 'Open': 'mean', 'Close': 'mean', 'Volume_(BTC)': 'sum', 'Volume_(Currency)': 'sum'})
    
    # create date object to return onlt rows starting from 2017
    date = pd.to_datetime("2017-01-01").date()
    return df.loc[date:]


def visualize_df(df):

    # plot the dataframe
    df.plot()

    # display legend
    plt.legend(loc='best')

    # display the plot
    plt.show()


# transform dataframe before plotting
df = transform_df(df)

# plot the dataframe
visualize_df(df)