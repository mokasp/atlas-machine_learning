#!/usr/bin/env python3
""""""


def array(df):
    selected = df[['High', 'Close']].tail(10)
    array = selected.to_numpy()
    return array