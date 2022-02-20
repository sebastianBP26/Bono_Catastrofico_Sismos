# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 16:47:14 2022

@author: Sebastian Barroso
"""

import pandas as pd

noaa = r'G:\Mi unidad\Tesis (Avances)\BASES\NOAA_12_02_2022.tsv'
df = pd.read_csv( noaa, encoding = 'unicode_escape', engine = 'python', sep = '\t')
df = df.iloc[1:]

df = df[['Year', 'Mo', 'Dy', 'Latitude', 'Longitude', 'Damage ($Mil)','Total Damage ($Mil)']]
a = df[~df['Total Damage ($Mil)'].isna()]
b = df[~df['Damage ($Mil)'].isna()]
