#!/usr/bin/env python3
""" module that contains function that sets the 'Timestamp' column as the
    index of the DataFrame."""


def index(df):
    """
        function that sets the 'Timestamp' column as the index of the
        DataFrame.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing a 'Timestamp' column.

        Returns:
        --------
        pandas.DataFrame
            The modified DataFrame with the 'Timestamp' column set as the
            index.

    """
    return df.set_index('Timestamp')
