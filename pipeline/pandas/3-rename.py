#!/usr/bin/env python3
""""""
import pandas as pd


def rename(df):
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df[['Datetime', 'Close']]