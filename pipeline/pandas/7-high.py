#!/usr/bin/env python3
""" module that contains a function that sorts the DataFrame by the
    'High' column in descending order."""


def high(df):
    """
       function that sorts the DataFrame by the 'High' column in
       descending order.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing at least a 'High' column.

        Returns:
        --------
        pandas.DataFrame
            The DataFrame sorted by the 'High' column in descending order.

    """
    return df.sort_values(by='High', ascending=False)
