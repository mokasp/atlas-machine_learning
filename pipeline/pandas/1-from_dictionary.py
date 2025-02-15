#!/usr/bin/env python3
""""""
import pandas as pd


diction = {'First': [0.0, 0.5, 1.0, 1.5], 'Second': ["one", "two", "three", "four"]}
num_idx = len(diction['First'])
idx = list(map(chr, range(65, 90)))[:num_idx]
df = pd.DataFrame(data=diction, index=idx)