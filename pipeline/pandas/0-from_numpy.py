#!/usr/bin/env python3
""" module containing function that coverts a 2D NumPy array to a
    Pandas DataFrame."""
import pandas as pd


def from_numpy(array):
    """
        Coverts a 2D NumPy array to a Pandas DataFrame. The columns are
        labeled with uppercase letters starting from 'A'. The number of
        columns in the DataFrame is based on the number of columns
        in the input array.

        Parameters:
        -----------
        array : numpy.ndarray
            A 2D NumPy array.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with data from the input array, with columns
            labeled 'A', 'B', etc.

    """
    num_columns = len(array[0])
    columns = list(map(chr, range(65, 90)))[:num_columns]
    dataframe = pd.DataFrame(data=array, columns=columns)
    return dataframe
