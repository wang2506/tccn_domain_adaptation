# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:46:45 2022

@author: henry
"""
import matplotlib.pyplot as plt
import pickle as pk
import os
import numpy as np

# from optim_utils.optim_parser import optim_parser

cwd = os.getcwd()

# args = 

# %% load data 
obj_f = '/obj_val/'
psi_f = '/psi_val/'
hat_ep_f = '/hat_ep_val/'

obj_vals = {}
psi_vals = {}
hat_ep_vals = {}
for i in range(1,3):   
    with open(cwd+obj_f+'init_bgap_st'+str(i),'rb') as f:
        obj_vals[i] = pk.load(f)

    with open(cwd+psi_f+'init_bgap_st'+str(i),'rb') as f:
        psi_vals[i] = pk.load(f)

    with open(cwd+hat_ep_f+'init_hat_ep'+str(i),'rb') as f:
        hat_ep_vals[i] = pk.load(f)
    
    psi_vals[i] = [np.round(j,0) for j in psi_vals[i][len(psi_vals[i].keys())-1]]
    
# %% plotting 
fig,ax = plt.subplots(figsize=(5,3))

ax = plt.plot(obj_vals[1],marker='x',linestyle='dashdot',c='darkblue',\
              label='Source Errors: '+str([np.round(i,2) for i in hat_ep_vals[1][:5]]))
ax = plt.plot(obj_vals[2],marker='x',linestyle='dotted',c='darkgreen',\
              label='Source Errors: '+str([np.round(i,2) for i in hat_ep_vals[2][:5]]))
plt.xlabel('Approximation Iteration')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Updates')

plt.legend() 

plt.grid(True)

plt.savefig(cwd+'/init_test_obj.png',bbox_inches='tight',dpi=1000)

# %% psi values
fig2,ax2 = plt.subplots(figsize=(5,3))


ax2 = plt.scatter(range(len(psi_vals[1])),psi_vals[1],marker='x',c='darkblue',\
              label='Source Errors: '+str([np.round(i,2) for i in hat_ep_vals[1][:5]]))
ax2 = plt.scatter(range(len(psi_vals[2])),psi_vals[2],marker='x',c='darkgreen',\
              label='Source Errors: '+str([np.round(i,2) for i in hat_ep_vals[2][:5]]))

plt.xlabel('Device No')
plt.ylabel('$\psi$')
plt.title('Source Target Status of the Networked Devices')

plt.legend() 

plt.grid(True)

plt.savefig(cwd+'/init_test_psi.png',bbox_inches='tight',dpi=1000)












