#!/usr/bin/env python3
""""""


def slice(df):
    extracted = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    return extracted.iloc[::60, :]