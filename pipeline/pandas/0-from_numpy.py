#!/usr/bin/env python3
""""""
import pandas as pd


def from_numpy(array):
    """"""
    num_columns = len(array[0])
    columns = list(map(chr, range(65, 90)))[:num_columns]
    dataframe = pd.DataFrame(data=array, columns=columns)
    return dataframe