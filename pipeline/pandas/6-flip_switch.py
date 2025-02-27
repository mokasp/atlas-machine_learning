#!/usr/bin/env python3
""" module that contains a function that sorts the data in reverse
    chronological order based on the 'Datetime' column, transposes the
    sorted DataFrame """


def flip_switch(df):
    """
        function that sorts the data in reverse chronological order based on
        the 'Datetime' column, transposes the sorted DataFrame, and returns
        the transformed DataFrame.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing at least a 'Datetime' column.

        Returns:
        --------
        pandas.DataFrame
            The sorted and transposed DataFrame.
    """
    sorted_df = df.sort_values(by='Timestamp', ascending=False)
    return sorted_df.transpose()
