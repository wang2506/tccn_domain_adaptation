# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:51:23 2022

@author: ch5b2
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk

cwd = os.getcwd()
# %% settings
dset_type = 'U'
# labels_type = 'iid'
labels_type = 'mild'
dset_split = 0
split_type = None

# %% extract data
taccs = {} 
raccs = {}
h1accs = {}
h2accs = {}
saccs = {}

for dset_type in ['M','U','S']:
    if dset_split == 0: # only one dataset
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_full_target','rb') as f:
            taccs[dset_type] = list((pk.load(f)).values())
    
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_full_rng','rb') as f:
            raccs[dset_type] = list((pk.load(f)).values())
        
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_full_h1','rb') as f:
            h1accs[dset_type] = list((pk.load(f)).values())
    
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_full_h2','rb') as f:
            h2accs[dset_type] = list((pk.load(f)).values())       
        
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_full_source','rb') as f:
            saccs[dset_type] = list((pk.load(f)).values())
    else:
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_full_target','rb') as f:
            taccs[dset_type] = list((pk.load(f)).values())
    
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_full_rng','rb') as f:
            raccs[dset_type] = list((pk.load(f)).values())
        
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_full_h1','rb') as f:
            h1accs[dset_type] = list((pk.load(f)).values())
    
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_full_h2','rb') as f:
            h2accs[dset_type] = list((pk.load(f)).values())       
        
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_full_source','rb') as f:
            saccs[dset_type] = list((pk.load(f)).values())

# %% plot bars for max-min
fig,ax = plt.subplots(1,3,figsize=(5,2),dpi=250,sharey=True)
cats = 4
width = 0.8
spacing = np.round(width/cats,2)
x = list(range(2))

tv = []
rv = []
h1v = []
h2v = []

for dset_type in ['M','U','S']:
    tv.append(np.max(taccs[dset_type]))
    rv.append(np.max(raccs[dset_type]))
    h1v.append(np.max(h1accs[dset_type]))
    h2v.append(np.max(h2accs[dset_type]))    
    
    tv.append(np.min(taccs[dset_type]))
    rv.append(np.min(raccs[dset_type]))  
    h1v.append(np.min(h1accs[dset_type]))
    h2v.append(np.min(h2accs[dset_type]))

ax[0].bar(x-1.5*spacing,tv[:2],width=spacing,\
        color='darkblue',edgecolor='black',label='Our Method')
ax[0].bar(x-0.5*spacing,rv[:2],width=spacing,\
        color='sienna',edgecolor='black',label='Random')
ax[0].bar(x+0.5*spacing,h1v[:2],width=spacing,\
        color='darkgreen',edgecolor='black',label='H1')
ax[0].bar(x+1.5*spacing,h2v[:2],width=spacing,\
        color='darkcyan',edgecolor='black',label='H2')
ax[0].set_ylabel('Accuracy (%)')
    
ax[1].bar(x-1.5*spacing,tv[2:4],width=spacing,\
        color='darkblue',edgecolor='black',label='Our Method')
ax[1].bar(x-0.5*spacing,rv[2:4],width=spacing,\
        color='sienna',edgecolor='black',label='Random')
ax[1].bar(x+0.5*spacing,h1v[2:4],width=spacing,\
        color='darkgreen',edgecolor='black',label='H1')
ax[1].bar(x+1.5*spacing,h2v[2:4],width=spacing,\
        color='darkcyan',edgecolor='black',label='H2')

ax[2].bar(x-1.5*spacing,tv[4:],width=spacing,\
        color='darkblue',edgecolor='black',label='Our Method')
ax[2].bar(x-0.5*spacing,rv[4:],width=spacing,\
        color='sienna',edgecolor='black',label='Random')
ax[2].bar(x+0.5*spacing,h1v[4:],width=spacing,\
        color='darkgreen',edgecolor='black',label='H1')
ax[2].bar(x+1.5*spacing,h2v[4:],width=spacing,\
        color='darkcyan',edgecolor='black',label='H2')

dset_vec = ['MNIST','USPS','SVHN']
for i in range(3):
    ax[i].set_xlabel(dset_vec[i])
    ax[i].set_xticklabels(['temp','Max','Min'])    
    ax[i].grid(True)
    ax[i].set_axisbelow(True)


h,l = ax[0].get_legend_handles_labels()
kw = dict(ncol=4,loc = 'lower center',frameon=False)
# kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
#(x, y, width, height)
# leg1 = ax2[0].legend(h[:3],l[:3],bbox_to_anchor=(-0.1,1.12,1.15,0.2),\
#                         mode='expand',fontsize=10,**kw2) #,**kw2)
leg1 = ax[0].legend(h,l,bbox_to_anchor=(-0.5,0.98,4,0.2),\
                        mode='expand',fontsize=10,**kw)
ax[0].add_artist(leg1)

fig.savefig(cwd+'/mt_plots/'+labels_type+'.png',dpi=1000,bbox_inches='tight')
fig.savefig(cwd+'/mt_plots/'+labels_type+'.pdf',dpi=1000,bbox_inches='tight')

