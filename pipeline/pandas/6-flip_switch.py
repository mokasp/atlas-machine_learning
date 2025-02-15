#!/usr/bin/env python3
""""""


def flip_switch(df):
    sorted_df = df.sort_values(by='Timestamp', ascending=False)
    return sorted_df.transpose()