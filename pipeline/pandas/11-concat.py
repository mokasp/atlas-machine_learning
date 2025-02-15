#!/usr/bin/env python3
""""""
import pandas as pd
index = __import__('10-index').index

def concat(df1, df2):
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:'1417411920']
    concatenated = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    return concatenated