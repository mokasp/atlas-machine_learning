#!/usr/bin/env python3
""" module that loads a dataset from a CSV file, performs data
    transformations, and visualizes the transformed data..

    Functions:
    ----------
    1. transform_df(df):
        -  performs various transformations and cleaning operations on the
            input DataFrame.
        - It drops unnecessary columns, renames the 'Timestamp' column,
            converts it to datetime format, handles missing data, and
            aggregates data by date.
        - It returns the transformed DataFrame containing data from 2017
            onwards.

    2. visualize_df(df):
        - This function generates a plot for the given DataFrame and
            displays it.
        - It ensures that the plot includes a legend and is shown to
            the user.

    Dependencies:
    ------------
    - pandas
    - matplotlib

    Usage:
    ------
    1. Load data from a CSV file into a pandas DataFrame using the
        `from_file` function.
    2. Apply the `transform_df()` function to clean and aggregate the data.
    3. Visualize the cleaned data using the `visualize_df()` function.

"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


def transform_df(df):
    """
        function that transforms the input DataFrame by cleaning and
        aggregating the data.

        The following operations are performed:
        - Drops the 'Weighted_Price' column.
        - Renames the 'Timestamp' column to 'Date' and converts the 'Date'
            column to datetime format.
        - Sets the 'Date' column as the index of the DataFrame.
        - Fills missing values in 'Close' column using forward filling.
        - Fills missing values in 'High', 'Low', and 'Open' columns with the
            corresponding value from the 'Close' column.
        - Fills missing values in 'Volume_(BTC)' and 'Volume_(Currency)'
            columns with 0.
        - Sorts the DataFrame by the 'Date' index.
        - Aggregates the data by date (using daily intervals) for columns:
            'High', 'Low', 'Open', 'Close', 'Volume_(BTC)', and
            'Volume_(Currency)'.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame containing historical cryptocurrency market
            data.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with cleaned and aggregated data starting from 2017
            onwards.

    """
    # drop weighted price column
    df = df.drop('Weighted_Price', axis=1)

    # rename the timestampe column to date and convert every column
    # entry to date format
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
    df[['Volume_(BTC)',
        'Volume_(Currency)']] = df[['Volume_(BTC)',
                                    'Volume_(Currency)']].fillna(0)

    # sort all rows by date
    df.sort_index()

    # aggregate data at daily intervals
    df = df.groupby('Date').agg({'High': 'max', 'Low': 'min', 'Open': 'mean',
                                 'Close': 'mean', 'Volume_(BTC)': 'sum',
                                 'Volume_(Currency)': 'sum'})

    # create date object to return onlt rows starting from 2017
    date = pd.to_datetime("2017-01-01").date()
    return df.loc[date:]


def visualize_df(df):
    """
        function that visualizes the input DataFrame by plotting the data.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame to be visualized.

        Returns:
        --------
        None

    """
    # plot the dataframe
    df.plot()

    # display legend
    plt.legend(loc='best')

    # display the plot
    plt.show()


# transform dataframe before plotting
df = transform_df(df)

# print the dataframe
print(df)

# plot the dataframe
visualize_df(df)
