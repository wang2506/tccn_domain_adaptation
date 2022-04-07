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
labels_type = 'mild'
dset_split = 0
# dset_split = 1
split_type = None
nn_style = 'MLP'

# %% extract data
taccs = {} 
raccs = {}
h1accs = {}
h2accs = {}
saccs = {}

if dset_split == 0: # only one dataset
    for dset_type in ['M']:#,'U','MM']:
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_target_nrg','rb') as f:
            our_nrg = pk.load(f)
    
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_rng_nrg','rb') as f:
            alpha_r_nrg = pk.load(f)
        
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_h1_nrg','rb') as f:
            alpha_h1_nrg = pk.load(f)
    
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_h2_nrg','rb') as f:
            alpha_h2_nrg = pk.load(f)

        # baselines for source/target split + alphas
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_rng_nrg','rb') as f:
            sta_r_nrg = pk.load(f)
        
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_h1_nrg','rb') as f:
            sta_h1_nrg = pk.load(f)
    
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_h2_nrg','rb') as f:
            sta_h2_nrg = pk.load(f)
else:
    for split_type in ['M+MM','M+U','MM+U']:
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_target_nrg','rb') as f:
            our_nrg = pk.load(f)
    
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_rng_nrg','rb') as f:
            alpha_r_nrg = pk.load(f)
        
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_h1_nrg','rb') as f:
            alpha_h1_nrg = pk.load(f)
    
        with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_h2_nrg','rb') as f:
            alpha_h2_nrg = pk.load(f)





# %% present/record the data to be used in the table



# %% save figures
# if dset_split == 0:
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'2.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'2.pdf',dpi=1000,bbox_inches='tight')
# else:
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.pdf',dpi=1000,bbox_inches='tight')    


