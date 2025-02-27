#!/usr/bin/env python3
""" module containing function that loads data from a file as a pandas
    dataframe """
import pandas as pd


def from_file(filename, delimiter):
    """
        Loads data from a file as a Pandas dataframe.
        
        Parameters:
        -----------
        filename: string
            name of csv file containing data to load
        deliminiter: string
            seperator for each entry

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with data loaded from file.

        """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
