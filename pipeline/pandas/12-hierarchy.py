#!/usr/bin/env python3
""" module containing function that rearranges the MultiIndex so that the
    'Timestamp' is the first level, the bitstamp and coinbase tables from
    timestamps, adds keys to the data, labeling rows from df2 as 'bitstamp'
    and rows from df1 as 'coinbase', and ensures the data is displayed in
    chronological order."""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
        function that rearranges the MultiIndex so that the 'Timestamp' is the
        first level, concatenates the bitstamp and coinbase tables from
        timestamps, adds keys to the data, labeling rows from df2 as
        'bitstamp' and rows from df1 as 'coinbase', and ensures the data is
        displayed in chronological order.

        Parameters:
        -----------
        df1 : pandas.DataFrame
            The first DataFrame (coinbase) containing a 'Timestamp' column.

        df2 : pandas.DataFrame
            The second DataFrame (bitstamp) containing a 'Timestamp' column.

        Returns:
        --------
        pandas.DataFrame
            The concatenated DataFrame with the MultiIndex reordered to show
            'Timestamp' as the first level, rows from df2 labeled as
            'bitstamp', and rows from df1 labeled as 'coinbase', sorted
            chronologically.

    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc['1417411980':'1417417980']
    df1 = df1.loc['1417411980':'1417417980']
    concatenated = pd.concat([df1, df2], keys=['coinbase', 'bitstamp'])
    return concatenated.reorder_levels([1, 0]).sort_index()
