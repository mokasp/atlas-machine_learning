#!/usr/bin/env python3
""" module containing function that emoves rows where the 'Close'
    column has NaN values."""


def prune(df):
    """
        function that emoves rows where the 'Close' column has NaN values.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing at least a 'Close' column.

        Returns:
        --------
        pandas.DataFrame
            The modified DataFrame with rows containing NaN values in the
            'Close' column removed.

    """
    return df.dropna(subset=['Close'])
