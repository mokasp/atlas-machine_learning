#!/usr/bin/env python3
""""""


def fill(df):
    """"""
    dropped = df.drop('Weighted_Price', axis=1)
    dropped['Close'] = dropped['Close'].fillna(method='ffill')
    dropped['High'] = dropped['High'].fillna(dropped['Close'])
    dropped['Low'] = dropped['Low'].fillna(dropped['Close'])
    dropped['Open'] = dropped['Open'].fillna(dropped['Close'])
    dropped[['Volume_(BTC)', 'Volume_(Currency)']] = dropped[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
    return dropped