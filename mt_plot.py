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
# dset_split = 1
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

if dset_split == 0: # only one dataset
    for idt,dset_type in enumerate(['M','U','MM']):
        for ids,seed in enumerate(seeds): 
            if nrg_mt == 0:
                acc_df = pd.read_csv(cwd+'/mt_results/'+dset_type+'/seed_'+str(seed)+'_'\
                        +labels_type \
                          +'_'+nn_style+'_acc.csv')
            else:
                acc_df = pd.read_csv(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e)+'_'\
                        +'seed_'+str(seed)+'_'+labels_type \
                          +'_'+nn_style+'_acc.csv')
            
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
else:
    for split_type in ['M+MM','M+U','MM+U']:
        if nrg_mt == 0:
            acc_df = pd.read_csv(cwd+'/mt_results/'+split_type+'/'+labels_type \
                      +'_'+nn_style+'_acc.csv')
        else:
            acc_df = pd.read_csv(cwd+'/mt_results/'+split_type+'/NRG'+str(phi_e)+'_'\
                    +labels_type \
                      +'_'+nn_style+'_acc.csv')
        taccs[split_type] = acc_df['ours'].tolist()
        raccs[split_type] = acc_df['rng'].tolist()
        h1accs[split_type] = acc_df['max_qty'].tolist()
        h2accs[split_type] = acc_df['unif_ratio'].tolist()
        saccs[split_type] = acc_df['source'].tolist()
        om_accs[split_type] = acc_df['o2m'].tolist()     

# # %% box-whisker plots
# fig,ax = plt.subplots(1,3,figsize=(5,2),dpi=250,sharey=True)
# # cats = 4
# cats = 5
# width = 0.8
# spacing = np.round(width/cats,2)
# x = list(range(3))

# tv = []
# rv = []
# h1v = []
# h2v = []
# omv = []

# if dset_split == 0:
#     dset_vec = ['MNIST','USPS','MNIST-M']
#     for idt,dset_type in enumerate(['M','U','MM']):
#         ax[idt].boxplot([taccs[dset_type],raccs[dset_type],h1accs[dset_type],\
#                 h2accs[dset_type]],\
#                 whis=1,sym='')
#         # ax[idt].boxplot(raccs[dset_type])
#         # ax[idt].boxplot(h1accs[dset_type])
#         # ax[idt].boxplot(h2accs[dset_type])

# %%
for idt,dset_type in enumerate(['M','U','MM']):
    ta_max[dset_type] = np.mean(ta_max[dset_type])
    ra_max[dset_type] = np.mean(ra_max[dset_type])
    h1_max[dset_type] = np.mean(h1_max[dset_type])
    h2_max[dset_type] = np.mean(h2_max[dset_type])
    om_max[dset_type] = np.mean(om_max[dset_type])
    
    ta_avg[dset_type] = np.mean(ta_avg[dset_type])
    ra_avg[dset_type] = np.mean(ra_avg[dset_type])
    h1_avg[dset_type] = np.mean(h1_avg[dset_type])
    h2_avg[dset_type] = np.mean(h2_avg[dset_type])
    om_avg[dset_type] = np.mean(om_avg[dset_type])    

    ta_min[dset_type] = np.mean(ta_min[dset_type])
    ra_min[dset_type] = np.mean(ra_min[dset_type])
    h1_min[dset_type] = np.mean(h1_min[dset_type])
    h2_min[dset_type] = np.mean(h2_min[dset_type])
    om_min[dset_type] = np.mean(om_min[dset_type])    

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
    for dset_type in ['M','U','MM']:
        # tv.append(np.max(taccs[dset_type]))
        # rv.append(np.max(raccs[dset_type]))
        # h1v.append(np.max(h1accs[dset_type]))
        # h2v.append(np.max(h2accs[dset_type]))    
        # omv.append(np.max(om_accs[dset_type]))
        
        # tv.append(np.average(taccs[dset_type]))
        # rv.append(np.average(raccs[dset_type]))
        # h1v.append(np.average(h1accs[dset_type]))
        # h2v.append(np.average(h2accs[dset_type]))    
        # omv.append(np.average(om_accs[dset_type]))
        
        # tv.append(np.min(taccs[dset_type]))
        # rv.append(np.min(raccs[dset_type]))  
        # h1v.append(np.min(h1accs[dset_type]))
        # h2v.append(np.min(h2accs[dset_type]))
        # omv.append(np.min(om_accs[dset_type]))
        tv.append(ta_max[dset_type]) #np.max(taccs[dset_type]))
        rv.append(ra_max[dset_type]) #np.max(raccs[dset_type]))
        h1v.append(h1_max[dset_type]) #np.max(h1accs[dset_type]))
        h2v.append(h2_max[dset_type]) #np.max(h2accs[dset_type]))    
        omv.append(om_max[dset_type]) #np.max(om_accs[dset_type]))
        
        tv.append(ta_avg[dset_type]) #np.average(taccs[dset_type]))
        rv.append(ra_avg[dset_type]) #np.average(raccs[dset_type]))
        h1v.append(h1_avg[dset_type]) #np.average(h1accs[dset_type]))
        h2v.append(h2_avg[dset_type]) #np.average(h2accs[dset_type]))    
        omv.append(om_avg[dset_type]) #np.average(om_accs[dset_type]))
        
        tv.append(ta_min[dset_type]) #np.min(taccs[dset_type]))
        rv.append(ra_min[dset_type]) #np.min(raccs[dset_type]))  
        h1v.append(h1_min[dset_type]) #np.min(h1accs[dset_type]))
        h2v.append(h2_min[dset_type]) #np.min(h2accs[dset_type]))
        omv.append(om_min[dset_type]) #np.min(om_accs[dset_type]))        
else:
    dset_vec = ['M+MM','M+U','MM+U']
    for split_type in ['M+MM','M+U','MM+U']:
        tv.append(np.max(taccs[split_type]))
        rv.append(np.max(raccs[split_type]))
        h1v.append(np.max(h1accs[split_type]))
        h2v.append(np.max(h2accs[split_type]))    
        omv.append(np.max(om_accs[split_type]))
        
        tv.append(np.average(taccs[split_type]))
        rv.append(np.average(raccs[split_type]))
        h1v.append(np.average(h1accs[split_type]))
        h2v.append(np.average(h2accs[split_type]))    
        omv.append(np.average(om_accs[split_type]))        
        
        tv.append(np.min(taccs[split_type]))
        rv.append(np.min(raccs[split_type]))  
        h1v.append(np.min(h1accs[split_type]))
        h2v.append(np.min(h2accs[split_type]))
        omv.append(np.min(om_accs[split_type]))

# ax[0].bar(x-2*spacing,tv[:3],width=spacing,\
#         color='darkblue',edgecolor='black',label=r'Our Method')
# ax[0].bar(x-1*spacing,rv[:3],width=spacing,\
#         color='sienna',edgecolor='black',label=r'Random-$\alpha$')
# ax[0].bar(x+0*spacing,h1v[:3],width=spacing,\
#         color='darkgreen',edgecolor='black',label='Qty-Scaled')
# ax[0].bar(x+1*spacing,h2v[:3],width=spacing,\
#         color='darkcyan',edgecolor='black',label='Uniform')
# ax[0].bar(x+2*spacing,omv[:3],width=spacing,\
#         color='purple',edgecolor='black',label='Avg-Degree')
# ax[0].set_ylabel('Accuracy (%)')

ax[0].bar([0],np.mean(taccs['M']),yerr=np.std(taccs['M']),ecolor='black',\
         capsize=5,width=width,\
        color='tab:blue',edgecolor='black',label=r'Our Method')
ax[0].bar([1],np.mean(raccs['M']),yerr=np.std(taccs['M']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:orange',edgecolor='black',label=r'Random-$\alpha$')
ax[0].bar([2],np.mean(h1accs['M']),yerr=np.std(h1accs['M']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:green',edgecolor='black',label=r'Qty-Scaled')    
ax[0].bar([3],np.mean(h2accs['M']),yerr=np.std(h2accs['M']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:brown',edgecolor='black',label=r'Uniform')  
ax[0].bar([4],np.mean(om_accs['M']),yerr=np.std(om_accs['M']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:purple',edgecolor='black',label=r'Avg-Degree')
ax[0].set_ylabel('Average Accuracy (%)')

ax[1].bar([0],np.mean(taccs['U']),yerr=np.std(taccs['U']),ecolor='black',\
         capsize=5,width=width,\
        color='tab:blue',edgecolor='black',label=r'Our Method')
ax[1].bar([1],np.mean(raccs['U']),yerr=np.std(taccs['U']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:orange',edgecolor='black',label=r'Random-$\alpha$')
ax[1].bar([2],np.mean(h1accs['U']),yerr=np.std(h1accs['U']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:green',edgecolor='black',label=r'Qty-Scaled')    
ax[1].bar([3],np.mean(h2accs['U']),yerr=np.std(h2accs['U']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:brown',edgecolor='black',label=r'Uniform')  
ax[1].bar([4],np.mean(om_accs['U']),yerr=np.std(om_accs['U']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:purple',edgecolor='black',label=r'Avg-Degree')

ax[2].bar([0],np.mean(taccs['MM']),yerr=np.std(taccs['MM']),ecolor='black',\
         capsize=5,width=width,\
        color='tab:blue',edgecolor='black',label=r'Our Method')
ax[2].bar([1],np.mean(raccs['MM']),yerr=np.std(taccs['MM']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:orange',edgecolor='black',label=r'Random-$\alpha$')
ax[2].bar([2],np.mean(h1accs['MM']),yerr=np.std(h1accs['MM']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:green',edgecolor='black',label=r'Qty-Scaled')    
ax[2].bar([3],np.mean(h2accs['MM']),yerr=np.std(h2accs['MM']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:brown',edgecolor='black',label=r'Uniform')  
ax[2].bar([4],np.mean(om_accs['MM']),yerr=np.std(om_accs['MM']),ecolor='black',\
          capsize=5,width=width,\
        color='tab:purple',edgecolor='black',label=r'Avg-Degree')

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
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'2.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'2.pdf',dpi=1000,bbox_inches='tight')
# else:
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.pdf',dpi=1000,bbox_inches='tight')    
