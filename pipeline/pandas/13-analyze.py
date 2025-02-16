#!/usr/bin/env python3
""""""


def analyze(df):
    """"""
    df = df.drop(columns='Timestamp')
    return df.describe()