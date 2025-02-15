#!/usr/bin/env python3
""""""


def prune(df):
    return df.dropna(subset=['Close'])