#!/usr/bin/env python3
""" module containing function that concatenates two DataFrames (df1 and df2)
    after indexing them on their 'Timestamp' columns. """
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
        function that concatenates two DataFrames (df1 and df2) after indexing
        them on their 'Timestamp' columns. Includes all rows from df2
        (bitstamp) up to and including timestamp 1417411920, and concatenates
        them on top of df1 (coinbase). Labels the rows from df2 as 'bitstamp'
        and the rows from df1 as 'coinbase'.

        Parameters:
        -----------
        df1 : pandas.DataFrame
            The first DataFrame (coinbase) containing a 'Timestamp' column.

        df2 : pandas.DataFrame
            The second DataFrame (bitstamp) containing a 'Timestamp' column.

        Returns:
        --------
        pandas.DataFrame
            The concatenated DataFrame with rows from df2 labeled as 'bitstamp'
            and rows from df1 labeled as 'coinbase'.

    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:'1417411920']
    concatenated = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    return concatenated
