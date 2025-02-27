#!/usr/bin/env python3
""" module containing a function that extracts the 'High', 'Low',
    'Close', and 'Volume_BTC' columns from the DataFrame, selects
    every 60th row from these columns"""


def slice(df):
    """
        function that extracts the 'High', 'Low', 'Close', and 'Volume_BTC'
        columns from the DataFrame, selects every 60th row from these columns,
        and returns the sliced DataFrame.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing at least the 'High', 'Low', 'Close', and
            'Volume_BTC' columns.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing every 60th row from the 'High', 'Low',
            'Close', and 'Volume_BTC' columns.

    """
    extracted = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    return extracted.iloc[::60, :]