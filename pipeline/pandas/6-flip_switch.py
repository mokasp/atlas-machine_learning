#!/usr/bin/env python3
""""""


def flip_switch(df):
    sorted_df = df.sort_values(by='Timestamp')
    return sorted_df.transpose()