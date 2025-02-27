#!/usr/bin/env python3
""" module containing function that removes the 'Weighted_Price' column, fills
    missing values in the 'Close' column with the previous row’s value, fills
    missing values in the 'High', 'Low', and 'Open' columns with the
    corresponding 'Close' value, and sets missing values in 'Volume_(BTC)' and
    'Volume_(Currency)' columns to 0. """


def fill(df):
    """
        function that removes the 'Weighted_Price' column, fills missing
        values in the 'Close' column with the previous row’s value, fills
        missing values in the 'High', 'Low', and 'Open' columns with the
        corresponding 'Close' value, and sets missing values in 'Volume_(BTC)'
        and 'Volume_(Currency)' columns to 0.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing at least the 'Weighted_Price', 'Close',
            'High', 'Low', 'Open', 'Volume_(BTC)', and 'Volume_(Currency)'
            columns.

        Returns:
        --------
        pandas.DataFrame
            The modified DataFrame after filling missing values and removing
            the 'Weighted_Price' column.

    """
    dropped = df.drop('Weighted_Price', axis=1)
    dropped['Close'] = dropped['Close'].fillna(method='ffill')
    dropped['High'] = dropped['High'].fillna(dropped['Close'])
    dropped['Low'] = dropped['Low'].fillna(dropped['Close'])
    dropped['Open'] = dropped['Open'].fillna(dropped['Close'])
    dropped[['Volume_(BTC)',
             'Volume_(Currency)']] = dropped[['Volume_(BTC)',
                                              'Volume_(Currency)']].fillna(0)
    return dropped
