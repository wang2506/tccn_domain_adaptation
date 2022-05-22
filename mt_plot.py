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
# %% settings
# labels_type = 'iid'
labels_type = 'mild'
dset_split = 0
dset_split = 1
split_type = None
nn_style = 'MLP'
nrg_mt = 1
phi_e = 1e1
# phi_e = 1e0
# phi_e = 1e-2 

seeds = [1,2,3,4,5]

# %% 
def extract_mma(dlist,max_list,min_list,avg_list):
    max_list.append(np.max(dlist))
    min_list.append(np.min(dlist))
    avg_list.append(np.average(dlist))
    return max_list,min_list,avg_list

# %% extract data
taccs,raccs,h1accs,h2accs,saccs,om_accs = {},{},{},{},{},{}
ta_max,ta_min,ta_avg = {},{},{} #[],[],[]
ra_max,ra_min,ra_avg = {},{},{}
h1_max,h1_min,h1_avg = {},{},{}
h2_max,h2_min,h2_avg = {},{},{}
s_max,s_min,s_avg = {},{},{}
om_max,om_min,om_avg = {},{},{}

for ids,seed in enumerate(seeds):
    if dset_split == 0:
        for idt,dset_type in enumerate(['M','U','MM']):
            if dset_type == 'MM':
                end = '_base_6'
            else:
                end = ''
            
            if nrg_mt == 0:
                acc_df = pd.read_csv(cwd+'/mt_results/'+dset_type+'/seed_'+str(seed)+'_'\
                        +labels_type \
                          +'_'+nn_style+end+'_acc.csv')                
            else:
                acc_df = pd.read_csv(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e)+'_'\
                        +'seed_'+str(seed)+'_'+labels_type \
                          +'_'+nn_style+end+'_acc.csv')
    
            if ids == 0:
                taccs[dset_type] = acc_df['ours'].tolist()
                raccs[dset_type] = acc_df['rng'].tolist()
                h1accs[dset_type] = acc_df['max_qty'].tolist()
                h2accs[dset_type] = acc_df['unif_ratio'].tolist()
                # saccs[dset_type] = acc_df['source'].tolist()
                om_accs[dset_type] = acc_df['o2m'].tolist()
                
                ta_max[dset_type],ta_min[dset_type],ta_avg[dset_type] = [],[],[]
                ra_max[dset_type],ra_min[dset_type],ra_avg[dset_type] = [],[],[]
                h1_max[dset_type],h1_min[dset_type],h1_avg[dset_type] = [],[],[]
                h2_max[dset_type],h2_min[dset_type],h2_avg[dset_type] = [],[],[]
                s_max[dset_type],s_min[dset_type],s_avg[dset_type] = [],[],[]
                om_max[dset_type],om_min[dset_type],om_avg[dset_type] = [],[],[]
            else:
                taccs[dset_type] += acc_df['ours'].tolist()
                raccs[dset_type] += acc_df['rng'].tolist()
                h1accs[dset_type] += acc_df['max_qty'].tolist()
                h2accs[dset_type] += acc_df['unif_ratio'].tolist()
                # saccs[dset_type] += acc_df['source'].tolist()
                om_accs[dset_type] += acc_df['o2m'].tolist()
            
            ta_max[dset_type],ta_min[dset_type],ta_avg[dset_type] \
                = extract_mma(acc_df['ours'].tolist(),ta_max[dset_type],ta_min[dset_type],ta_avg[dset_type])
            ra_max[dset_type],ra_min[dset_type],ra_avg[dset_type] \
                = extract_mma(acc_df['rng'].tolist(),ra_max[dset_type],ra_min[dset_type],ra_avg[dset_type])
            h1_max[dset_type],h1_min[dset_type],h1_avg[dset_type] \
                = extract_mma(acc_df['max_qty'].tolist(),h1_max[dset_type],h1_min[dset_type],h1_avg[dset_type])
            h2_max[dset_type],h2_min[dset_type],h2_avg[dset_type] \
                = extract_mma(acc_df['unif_ratio'].tolist(),h2_max[dset_type],h2_min[dset_type],h2_avg[dset_type])
            # s_max[dset_type],s_min[dset_type],s_avg[dset_type] \
            #     = extract_mma(acc_df['source'].tolist(),s_max[dset_type],s_min[dset_type],s_avg[dset_type])
            om_max[dset_type],om_min[dset_type],om_avg[dset_type] \
                = extract_mma(acc_df['o2m'].tolist(),om_max[dset_type],om_min[dset_type],om_avg[dset_type])              
            
    elif dset_split == 1 :
        for split_type in ['M+MM','M+U','MM+U']:    
            if 'MM' in split_type:
                end = '_base_6'
            else:
                end = ''
                
            if nrg_mt == 0:
                acc_df = pd.read_csv(cwd+'/mt_results/'+split_type+'/seed_'+str(seed)+'_'\
                        +labels_type \
                          +'_'+nn_style+end+'_acc.csv')
            else:
                acc_df = pd.read_csv(cwd+'/mt_results/'+split_type+'/NRG'+str(phi_e)+'_'\
                        +'seed_'+str(seed)+'_'+labels_type \
                          +'_'+nn_style+end+'_acc.csv')    
  
            if ids == 0:
                taccs[split_type] = acc_df['ours'].tolist()
                raccs[split_type] = acc_df['rng'].tolist()
                h1accs[split_type] = acc_df['max_qty'].tolist()
                h2accs[split_type] = acc_df['unif_ratio'].tolist()
                # saccs[split_type] = acc_df['source'].tolist()
                om_accs[split_type] = acc_df['o2m'].tolist()
                
                ta_max[split_type],ta_min[split_type],ta_avg[split_type] = [],[],[]
                ra_max[split_type],ra_min[split_type],ra_avg[split_type] = [],[],[]
                h1_max[split_type],h1_min[split_type],h1_avg[split_type] = [],[],[]
                h2_max[split_type],h2_min[split_type],h2_avg[split_type] = [],[],[]
                s_max[split_type],s_min[split_type],s_avg[split_type] = [],[],[]
                om_max[split_type],om_min[split_type],om_avg[split_type] = [],[],[]
            else:
                taccs[split_type] += acc_df['ours'].tolist()
                raccs[split_type] += acc_df['rng'].tolist()
                h1accs[split_type] += acc_df['max_qty'].tolist()
                h2accs[split_type] += acc_df['unif_ratio'].tolist()
                # saccs[split_type] += acc_df['source'].tolist()
                om_accs[split_type] += acc_df['o2m'].tolist()
            
            ta_max[split_type],ta_min[split_type],ta_avg[split_type] \
                = extract_mma(acc_df['ours'].tolist(),ta_max[split_type],ta_min[split_type],ta_avg[split_type])
            ra_max[split_type],ra_min[split_type],ra_avg[split_type] \
                = extract_mma(acc_df['rng'].tolist(),ra_max[split_type],ra_min[split_type],ra_avg[split_type])
            h1_max[split_type],h1_min[split_type],h1_avg[split_type] \
                = extract_mma(acc_df['max_qty'].tolist(),h1_max[split_type],h1_min[split_type],h1_avg[split_type])
            h2_max[split_type],h2_min[split_type],h2_avg[split_type] \
                = extract_mma(acc_df['unif_ratio'].tolist(),h2_max[split_type],h2_min[split_type],h2_avg[split_type])
            # s_max[split_type],s_min[split_type],s_avg[split_type] \
            #     = extract_mma(acc_df['source'].tolist(),s_max[split_type],s_min[split_type],s_avg[split_type])
            om_max[split_type],om_min[split_type],om_avg[split_type] \
                = extract_mma(acc_df['o2m'].tolist(),om_max[split_type],om_min[split_type],om_avg[split_type])

# %% plot bars for max-min
fig,ax = plt.subplots(1,3,figsize=(5,2),dpi=250,sharey=True)
# cats = 4
cats = 5
width = 0.8
spacing = np.round(width/cats,2)
# x = list(range(3))
x = list(range(5))

tv = []
rv = []
h1v = []
h2v = []
omv = []

if dset_split == 0:
    dset_vec = ['MNIST','USPS','MNIST-M']
    for i,j in enumerate(['M','U','MM']):
        ax[i].bar([0],np.mean(taccs[j]),yerr=np.std(taccs[j]),ecolor='black',\
                 capsize=5,width=width,\
                color='tab:blue',edgecolor='black',label=r'Our Method')
        ax[i].bar([1],np.mean(raccs[j]),yerr=np.std(taccs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:orange',edgecolor='black',label=r'Random-$\alpha$')
        ax[i].bar([2],np.mean(h1accs[j]),yerr=np.std(h1accs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:green',edgecolor='black',label=r'Qty-Scaled')    
        ax[i].bar([3],np.mean(h2accs[j]),yerr=np.std(h2accs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:brown',edgecolor='black',label=r'Uniform')  
        ax[i].bar([4],np.mean(om_accs[j]),yerr=np.std(om_accs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:purple',edgecolor='black',label=r'Avg-Degree')
        if i == 0: 
            ax[i].set_ylabel('Average Accuracy (%)')
else:
    dset_vec = ['M+MM','M+U','MM+U']
    for i,j in enumerate(dset_vec):
        ax[i].bar([0],np.mean(taccs[j]),yerr=np.std(taccs[j]),ecolor='black',\
                 capsize=5,width=width,\
                color='tab:blue',edgecolor='black',label=r'Our Method')
        ax[i].bar([1],np.mean(raccs[j]),yerr=np.std(taccs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:orange',edgecolor='black',label=r'Random-$\alpha$')
        ax[i].bar([2],np.mean(h1accs[j]),yerr=np.std(h1accs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:green',edgecolor='black',label=r'Qty-Scaled')    
        ax[i].bar([3],np.mean(h2accs[j]),yerr=np.std(h2accs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:brown',edgecolor='black',label=r'Uniform')  
        ax[i].bar([4],np.mean(om_accs[j]),yerr=np.std(om_accs[j]),ecolor='black',\
                  capsize=5,width=width,\
                color='tab:purple',edgecolor='black',label=r'Avg-Degree')
        if i == 0: 
            ax[i].set_ylabel('Average Accuracy (%)')


for i in range(3):
    ax[i].set_xlabel(dset_vec[i])
    # ax[i].set_xticks(range(3))
    # ax[i].set_xticklabels(['Max','Avg','Min'])    
    ax[i].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
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
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_avg.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_avg.pdf',dpi=1000,bbox_inches='tight')
# else:
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.pdf',dpi=1000,bbox_inches='tight')    

