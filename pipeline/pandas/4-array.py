#!/usr/bin/env python3
""" a module containing a function that selects the last 10 rows of the 'High'
    and 'Close' columns from the DataFrame, and converts these selected values
    into a NumPy ndarrayst"""


def array(df):
    """
        function that selects the last 10 rows of the 'High' and 'Close'
        columns from the DataFrame, converts these selected values into a
        NumPy ndarray, and returns the result.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing at least the 'High' and 'Close' columns.

        Returns:
        --------
        numpy.ndarray
            A 2D NumPy array containing the last 10 rows of the 'High' and
            'Close' columns.

    """
    selected = df[['High', 'Close']].tail(10)
    array = selected.to_numpy()
    return array
