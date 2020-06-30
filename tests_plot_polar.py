# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:39:29 2020

@author: n052328
"""
import matplotlib.pyplot as plt
import numpy as np

#%% Compute areas and colors
N = 150
N = kkcum.shape[0]
r = 2 * np.random.rand(N)
r = 200 * kkcum
theta = 2 * np.pi * np.random.rand(N)
theta = 2 * np.pi * np.linspace(0.0,1.0, kkcum.shape[0], endpoint=False)
area = 200 * r**2
colors = theta
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
#c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
c = ax.scatter(theta, 100*kkcum, c=colors, s=area, cmap='hsv', alpha=0.75)

