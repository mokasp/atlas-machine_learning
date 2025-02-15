#!/usr/bin/env python3
""""""


def fill(df):
    """"""
    dropped = df.drop('Weighted_Price', axis=1)
    dropped['Close'] = dropped['Close'].fillna(method='ffill')
    dropped[['High', 'Low', 'Open']] = dropped[['High', 'Low', 'Open']].fillna(dropped['Close'])
    dropped[['Volume_(BTC)', 'Volume_(Currency)']] = dropped[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
    return dropped