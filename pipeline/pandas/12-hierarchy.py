#!/usr/bin/env python3
""""""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc['1417411980':'1417417980']
    df1 = df1.loc['1417411980':'1417417980']
    concatenated = pd.concat([df1, df2], keys=['coinbase', 'bitstamp'])
    return concatenated.reorder_levels([1, 0]).sort_index()