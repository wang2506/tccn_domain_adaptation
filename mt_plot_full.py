# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:51:23 2022

@author: ch5b2
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import pandas as pd

cwd = os.getcwd()
# %% Some notes
# This compares our optimization's source/target determination
# against some heuristic baselines.
#
# The file that compares our weight determination only is in mt_plot.py
# %% settings
# labels_type = 'iid'
labels_type = 'mild'
dset_split = 0
dset_split = 1
nrg_mt = 1
split_type = None
nn_style = 'MLP'
phi_e = 1e1

# %% extract data
taccs = {} 
raccs = {}
h1accs = {}
h2accs = {}
saccs = {}
oo_accs = {} 

if dset_split == 0: # only one dataset
    for dset_type in ['M','U','MM']:
        if nrg_mt == 0:
            acc_df1 = pd.read_csv(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                      +'_'+nn_style+'_acc.csv')
            acc_df2 = pd.read_csv(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                      +'_'+nn_style+'_acc.csv')                
        else:
            acc_df1 = pd.read_csv(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e)+'_'\
                    +labels_type \
                      +'_'+nn_style+'_acc.csv')
            acc_df2 = pd.read_csv(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e)\
                    +'_st_det_'+labels_type \
                      +'_'+nn_style+'_acc.csv')
        taccs[dset_type] = acc_df1['ours'].tolist()
        raccs[dset_type] = acc_df2['rng'].dropna().tolist()
        h1accs[dset_type] = acc_df2['geq_avg_acc'].dropna().tolist()
        h2accs[dset_type] = acc_df2['max_acc'].tolist()
        oo_accs[dset_type] = acc_df1['o2o'].tolist()                   
else:
    for split_type in ['M+MM','M+U','MM+U']:
        if nrg_mt == 0:
            acc_df1 = pd.read_csv(cwd+'/mt_results/'+split_type+'/'+labels_type \
                      +'_'+nn_style+'_acc.csv')
            acc_df2 = pd.read_csv(cwd+'/mt_results/'+split_type+'/st_det_'+labels_type \
                      +'_'+nn_style+'_acc.csv')                
        else:
            acc_df1 = pd.read_csv(cwd+'/mt_results/'+split_type+'/NRG'+str(phi_e)+'_'\
                    +labels_type \
                      +'_'+nn_style+'_acc.csv')
            acc_df2 = pd.read_csv(cwd+'/mt_results/'+split_type+'/NRG'+str(phi_e)\
                    +'_st_det_'+labels_type \
                      +'_'+nn_style+'_acc.csv')
        taccs[split_type] = acc_df1['ours'].tolist()
        raccs[split_type] = acc_df2['rng'].dropna().tolist()
        h1accs[split_type] = acc_df2['geq_avg_acc'].dropna().tolist()
        h2accs[split_type] = acc_df2['max_acc'].tolist()
        oo_accs[split_type] = acc_df1['o2o'].tolist()  

# %% plot bars for max-min
fig,ax = plt.subplots(1,3,figsize=(5,2),dpi=250,sharey=True)
# cats = 4
cats = 5
width = 0.8
spacing = np.round(width/cats,2)
x = list(range(3))

tv = []
rv = []
h1v = []
h2v = []
oov = []

if dset_split == 0:
    dset_vec = ['MNIST','USPS','MNIST-M']
    for dset_type in ['M','U','MM']:
        tv.append(np.max(taccs[dset_type]))
        rv.append(np.max(raccs[dset_type]))
        h1v.append(np.max(h1accs[dset_type]))
        h2v.append(np.max(h2accs[dset_type]))    
        oov.append(np.max(oo_accs[dset_type]))    
        
        tv.append(np.average(taccs[dset_type]))
        rv.append(np.average(raccs[dset_type]))
        h1v.append(np.average(h1accs[dset_type]))
        h2v.append(np.average(h2accs[dset_type]))    
        oov.append(np.average(oo_accs[dset_type]))   
        
        tv.append(np.min(taccs[dset_type]))
        rv.append(np.min(raccs[dset_type]))  
        h1v.append(np.min(h1accs[dset_type]))
        h2v.append(np.min(h2accs[dset_type]))
        oov.append(np.min(oo_accs[dset_type]))   
else:
    dset_vec = ['M+MM','M+U','MM+U']
    for split_type in ['M+MM','M+U','MM+U']:
        tv.append(np.max(taccs[split_type]))
        rv.append(np.max(raccs[split_type]))
        h1v.append(np.max(h1accs[split_type]))
        h2v.append(np.max(h2accs[split_type]))    
        oov.append(np.max(oo_accs[split_type])) 
        
        tv.append(np.average(taccs[split_type]))
        rv.append(np.average(raccs[split_type]))
        h1v.append(np.average(h1accs[split_type]))
        h2v.append(np.average(h2accs[split_type]))    
        oov.append(np.average(oo_accs[split_type]))   
        
        tv.append(np.min(taccs[split_type]))
        rv.append(np.min(raccs[split_type]))  
        h1v.append(np.min(h1accs[split_type]))
        h2v.append(np.min(h2accs[split_type]))
        oov.append(np.min(oo_accs[split_type]))  

ax[0].bar(x-2*spacing,tv[:3],width=spacing,\
        color='darkblue',edgecolor='black',label='Our Method')
ax[0].bar(x-1*spacing,rv[:3],width=spacing,\
        color='sienna',edgecolor='black',label=r'Random-$\psi$')
ax[0].bar(x+0*spacing,h1v[:3],width=spacing,\
        color='darkgreen',edgecolor='black',label='Geq-Max-Acc')
ax[0].bar(x+1*spacing,h2v[:3],width=spacing,\
        color='darkcyan',edgecolor='black',label='Only-Max-Acc')
ax[0].bar(x+2*spacing,oov[:3],width=spacing,\
        color='purple',edgecolor='black',label='Single Matching')    
ax[0].set_ylabel('Accuracy (%)')

ax[1].bar(x-2*spacing,tv[3:6],width=spacing,\
        color='darkblue',edgecolor='black',label='Our Method')
ax[1].bar(x-1*spacing,rv[3:6],width=spacing,\
        color='sienna',edgecolor='black',label=r'Random-$\psi$')
ax[1].bar(x+0*spacing,h1v[3:6],width=spacing,\
        color='darkgreen',edgecolor='black',label='Geq-Max-Acc')
ax[1].bar(x+1*spacing,h2v[3:6],width=spacing,\
        color='darkcyan',edgecolor='black',label='Only-Max-Acc')
ax[1].bar(x+2*spacing,oov[3:6],width=spacing,\
        color='purple',edgecolor='black',label='Single Matching')    
    
ax[2].bar(x-2*spacing,tv[6:],width=spacing,\
        color='darkblue',edgecolor='black',label='Our Method')
ax[2].bar(x-1*spacing,rv[6:],width=spacing,\
        color='sienna',edgecolor='black',label=r'Random-$\psi$')
ax[2].bar(x+0*spacing,h1v[6:],width=spacing,\
        color='darkgreen',edgecolor='black',label='Geq-Max-Acc')
ax[2].bar(x+1*spacing,h2v[6:],width=spacing,\
        color='darkcyan',edgecolor='black',label='Only-Max-Acc')
ax[2].bar(x+2*spacing,oov[6:],width=spacing,\
        color='purple',edgecolor='black',label='Single Matching')    
    
for i in range(3):
    ax[i].set_xlabel(dset_vec[i])
    ax[i].set_xticks(range(3))
    ax[i].set_xticklabels(['Max','Avg','Min'])    
    ax[i].grid(True)
    ax[i].set_axisbelow(True)

h,l = ax[0].get_legend_handles_labels()
kw = dict(ncol=2,loc = 'lower center',frameon=False)
kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax[0].legend(h[:3],l[:3],bbox_to_anchor=(-0.5,1.12,4,0.2),\
                        mode='expand',fontsize=10,**kw2) #,**kw2)
leg2 = ax[0].legend(h[3:],l[3:],bbox_to_anchor=(-0.5,0.98,4,0.2),\
                        mode='expand',fontsize=10,**kw)
ax[0].add_artist(leg1)

# %% save figures
# if dset_split == 0:
#     fig.savefig(cwd+'/mt_plots/st_det_'+labels_type+'2.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/st_det_'+labels_type+'2.pdf',dpi=1000,bbox_inches='tight')
# else:
#     fig.savefig(cwd+'/mt_plots/st_det_'+labels_type+'_mixed.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/st_det_'+labels_type+'_mixed.pdf',dpi=1000,bbox_inches='tight')    


