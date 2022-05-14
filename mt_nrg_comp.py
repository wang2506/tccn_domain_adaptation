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

phi_e_min = 1e-2
phi_e_max = 1e2

if dset_split == 0: # only one dataset
    for dset_type in ['M']:#,'U','MM']:
        ## our method needs the phi_e spec'd
        with open(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e_min)+'_'\
                  +labels_type \
                  +'_'+nn_style\
                +'_target_nrg','rb') as f:
            our_nrg_max = np.round(pk.load(f),2)
    
        with open(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e_max)+'_'\
                  +labels_type \
                  +'_'+nn_style\
                +'_target_nrg','rb') as f:
            our_nrg_min = np.round(pk.load(f),2)   
        
        with open(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e_min)+'_'\
                  +labels_type \
                  +'_'+nn_style\
                +'_full_target','rb') as f:
            our_acc_max = pk.load(f)
        our_acc_max = np.round(np.mean(list(our_acc_max.values())),2)
    
        with open(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e_max)+'_'\
                  +labels_type \
                  +'_'+nn_style\
                +'_full_target','rb') as f:
            our_acc_min = pk.load(f)            
        our_acc_min = np.round(np.mean(list(our_acc_min.values())),2)
        
        ## no need for phi_e spec
        # baselines for alphas
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_rng_nrg','rb') as f:
            alpha_r_nrg = np.round(pk.load(f),2)
        
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_h1_nrg','rb') as f:
            alpha_h1_nrg = np.round(pk.load(f),2)
    
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_h2_nrg','rb') as f:
            alpha_h2_nrg = np.round(pk.load(f),2)

        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_full_rng','rb') as f:
            alpha_r_acc = pk.load(f)
        alpha_r_acc = np.round(np.mean(list(alpha_r_acc.values())),2)
        
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_full_h1','rb') as f:
            alpha_h1_acc = pk.load(f)
        alpha_h1_acc = np.round(np.mean(list(alpha_h1_acc.values())),2)
        
        with open(cwd+'/mt_results/'+dset_type+'/'+labels_type \
                  +'_'+nn_style\
                +'_full_h2','rb') as f:
            alpha_h2_acc = pk.load(f)
        alpha_h2_acc = np.round(np.mean(list(alpha_h2_acc.values())),2)

        ## baselines for source/target split + alphas
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_full_rng_nrg','rb') as f:
            sta_r_nrg = np.round(pk.load(f),2)
        
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_full_h1_nrg','rb') as f:
            sta_h1_nrg = np.round(pk.load(f),2)
    
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_full_h2_nrg','rb') as f:
            sta_h2_nrg = np.round(pk.load(f),2)
            
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_full_rng','rb') as f:
            sta_r_acc = pk.load(f)
        sta_r_acc = np.round(np.mean(list(sta_r_acc.values())),2)
        
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_full_h1','rb') as f:
            sta_h1_acc = pk.load(f)
        sta_h1_acc = np.round(np.mean(list(sta_h1_acc.values())),2)
    
        with open(cwd+'/mt_results/'+dset_type+'/st_det_'+labels_type \
                  +'_'+nn_style\
                +'_full_h2','rb') as f:
            sta_h2_acc = pk.load(f)      
        sta_h2_acc = np.round(np.mean(list(sta_h2_acc.values())),2)
# else:
#     for split_type in ['M+MM','M+U','MM+U']:
#         with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
#                   +'_'+nn_style\
#                 +'_target_nrg','rb') as f:
#             our_nrg = pk.load(f)
    
#         with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
#                   +'_'+nn_style\
#                 +'_rng_nrg','rb') as f:
#             alpha_r_nrg = pk.load(f)
        
#         with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
#                   +'_'+nn_style\
#                 +'_h1_nrg','rb') as f:
#             alpha_h1_nrg = pk.load(f)
    
#         with open(cwd+'/mt_results/'+split_type+'/'+labels_type \
#                   +'_'+nn_style\
#                 +'_h2_nrg','rb') as f:
#             alpha_h2_nrg = pk.load(f)


# %% present/record the data to be used in the table
tdf = pd.DataFrame()
tdf['Row_Labels'] = ['Accuracy','Energy']

tdf['Ours_Max'] = [our_acc_max,our_nrg_max]
tdf['Ours_Min'] = [our_acc_min,our_nrg_min]

tdf['Alpha_Rng'] = [alpha_r_acc,alpha_r_nrg]
tdf['ST_Rng'] = [sta_r_acc,sta_r_nrg]

tdf['Alpha_H1'] = [alpha_h1_acc,alpha_h1_nrg]
tdf['ST_H1'] = [sta_h1_acc,sta_h1_nrg]

tdf['Alpha_H2'] = [alpha_h2_acc,alpha_h2_nrg]
tdf['ST_H2'] = [sta_h2_acc,sta_h2_nrg]

tdf.to_csv(cwd+'/mt_nrg_results/'+dset_type+'_NRG_df.csv')


# %% save figures
# if dset_split == 0:
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'2.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'2.pdf',dpi=1000,bbox_inches='tight')
# else:
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.png',dpi=1000,bbox_inches='tight')
#     fig.savefig(cwd+'/mt_plots/'+labels_type+'_mixed.pdf',dpi=1000,bbox_inches='tight')    


