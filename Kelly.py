# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:09:52 2020

@author: n052328
"""

import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import Generator, PCG64
#%% estudio de movimiento Browniano
plt.close('all')
rg = Generator(PCG64())
rg.standard_normal()
# df = pd.DataFrame(rg.normal(loc=[0.0002,0.0002,0.0002,0.0002,0.0002],
#                          scale=[.002,.003,.004,.005,.006],
#                          size=(1000,5)),
#                index=pd.date_range('1/1/2000', periods=1000))
df = pd.DataFrame(rg.normal(loc=[0.0002]*25,
                         scale=[.009]*25,
                         size=(1000,25)),
               index=pd.date_range('1/1/2000', periods=1000))
df.values.mean(),df.values.std()
df.describe()
df = df.cumsum()
df.plot()
df[-1:].mean(axis=1),df[-1:].std(axis=1)/25**.5
(.0002 - .5*(.009**2))*1000
