#!/usr/bin/env python3
""" module containing function that computes descriptive statistics for all
    columns except the 'Timestamp' column."""


def analyze(df):
    """
        function that computes descriptive statistics for all columns except
        the 'Timestamp' column.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing the data, with a 'Timestamp' column.

        Returns:
        --------
        pandas.DataFrame
            A new DataFrame containing descriptive statistics (count, mean,
            std, min, 25%, 50%, 75%, max) for all columns except 'Timestamp'.

    """
    df = df.drop(columns='Timestamp')
    return df.describe()
