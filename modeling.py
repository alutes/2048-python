#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:35:19 2023

@author: alutes
"""

from pathlib import Path
base_path = Path("/Users/alutes/Documents/Data/")
files = [f for f in base_path.glob('**/*.csv') if f.is_file()]
df = pd.concat([pd.read_csv(f) for f in files])
del df['Unnamed: 0']

def string_to_mat(string):
    return np.matrix(string).reshape([4,4])

df['mat'] = df.game.apply(string_to_mat)

mat = df.iloc[0]['mat']

