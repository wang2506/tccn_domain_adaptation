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
pwd = os.path.dirname(cwd)
# args = 

# %% load data 
obj_f = '/obj_val/'
psi_f = '/psi_val/'
hat_ep_f = '/hat_ep_val/'
alpha_f = '/alpha_val/'

obj_vals = {}
psi_vals = {}
hat_ep_vals = {}
alpha_vals = {}

with open(cwd+'/ablation_study/unif_maxdiv_pairs','rb') as f:
    div_pairs = pk.load(f)

with open(pwd+'/optim_results/'+alpha_f+'init_alpha1','rb') as f:
    tc_alpha = pk.load(f)

with open(cwd+alpha_f+'init_alpha_div1','rb') as f:
    tc_alpha_div = pk.load(f)

with open(cwd+psi_f+'init_bgap_st'+str(1),'rb') as f:
    psi_vals[1] = pk.load(f)
psi_vals[1] = [int(np.round(j,0)) for j in psi_vals[1][len(psi_vals[1].keys())-1]]

# tp_alpha = tc_alpha[:5,5:]
sources = np.where(np.array(psi_vals[1])==0)
targets = np.where(np.array(psi_vals[1])==1)

tp_alpha = tc_alpha[sources][:,targets]
tp_alpha2 = [ta[0].tolist() for ta in tp_alpha]
for tai,taj in enumerate(tp_alpha2):
    for ttai,ttaj in enumerate(taj):
        taj[ttai] = np.round(ttaj,2)

tc_alpha_div = tc_alpha_div[sources][:,targets]
tc_alpha_div2 = [ta[0].tolist() for ta in tc_alpha_div]
for tai,taj in enumerate(tc_alpha_div2):
    for ttai,ttaj in enumerate(taj):
        taj[ttai] = np.round(ttaj,2)

# %% obj_fxn conv check
fig,ax = plt.subplots(figsize=(5,3))

width = 1
cats = 5*2
x = np.arange(len(tp_alpha2[0]))

t_pairs = div_pairs[sources[0],targets[0]]

ax.bar(x-4.5*width/cats,tp_alpha2[0],width=width/cats,\
       color='darkblue',edgecolor='black')#,label=str(t_pairs))
ax.bar(x-3.5*width/cats,tc_alpha_div2[0],width=width/cats,\
       color='royalblue',edgecolor='black')

ax.bar(x-2.5*width/cats,tp_alpha2[1],width=width/cats,\
       color='darkgreen',edgecolor='black')
ax.bar(x-1.5*width/cats,tc_alpha_div2[1],width=width/cats,\
       color='green',edgecolor='black')

ax.bar(x-0.5*width/cats,tp_alpha2[2],width=width/cats,\
       color='darkmagenta',edgecolor='black')
ax.bar(x+0.5*width/cats,tc_alpha_div2[2],width=width/cats,\
       color='magenta',edgecolor='black')

ax.bar(x+1.5*width/cats,tp_alpha2[3],width=width/cats,\
       color='darkgoldenrod',edgecolor='black')
ax.bar(x+2.5*width/cats,tc_alpha_div2[3],width=width/cats,\
       color='gold',edgecolor='black')

ax.bar(x+3.5*width/cats,tp_alpha2[4],width=width/cats,\
       color='darkcyan',edgecolor='black')
ax.bar(x+4.5*width/cats,tc_alpha_div2[4],width=width/cats,\
       color='cyan',edgecolor='black')

ax.set_xlabel('Source Device')
ax.set_xticklabels(['0','1','2','3','4','5'])

ax.set_ylabel('Alpha TO Target Device')
ax.set_title('Effect of Divergence Estimates on Alpha Value')

# plt.legend()

plt.grid(True)

# plt.savefig(cwd+'/init_test_alpha.png',bbox_inches='tight',dpi=1000)

# %% alpha values









