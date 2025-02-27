#!/usr/bin/env python3
""" module containing function that renames and converts a column in a
    Pandas dataframe to Datetime """
import pandas as pd


def rename(df):
    """ 
        rename the 'Timestamp' column to 'Datetime', convert the timestamp
        values to datetime format, and display only the 'Datetime' and
        'Close' columns.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing at least the 'Timestamp' and
            'Close' columns.

        Returns:
        --------
        pandas.DataFrame
            A modified DataFrame with the 'Timestamp' column renamed to
            'Datetime', the 'Timestamp' values converted to datetime, and
            only 'Datetime' and 'Close' columns displayed.
    """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]