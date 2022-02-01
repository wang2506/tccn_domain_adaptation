# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:46:45 2022

@author: henry
"""
import matplotlib.pyplot as plt
import pickle as pk
import os
import numpy as np

cwd = os.getcwd()

# %% load data 
subfolder = '/obj_val/'
with open(cwd+subfolder+'initial_test','rb') as f:
    obj_vals = pk.load(f)

# %% plotting 
fig,ax = plt.subplots(figsize=(5,3))

ax = plt.plot(obj_vals,marker='x',linestyle='dashdot')
plt.xlabel('Approximation iteration')
plt.ylabel('Objective function value')
plt.title('Objective function updates')

plt.grid(True)

plt.savefig(cwd+subfolder+'init_test.png',bbox_inches='tight',dpi=1000)

# %% 
# fig2,ax2 = plt.subplots(figsize=(5,3))

# ax2 = plt












