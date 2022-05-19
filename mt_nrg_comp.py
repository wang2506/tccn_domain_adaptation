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
from copy import deepcopy

cwd = os.getcwd()
# %% settings
labels_type = 'mild'
dset_split = 0
# dset_split = 1
split_type = None
nn_style = 'MLP'

nrg_mt = 1
phi_e = 1e1

pd.set_option('display.max_columns', None)
# %% extract data
seeds = [1,2,3,4,5]
# seeds = [1]
dsets = ['M','U','MM']
# dsets = ['M']

nrg_per_dset = pd.DataFrame()
cols = ['ours','rng']
nrg_per_dset['ours'] = [0,0]

for dset_type in dsets: 
    for seed in seeds:
        if nrg_mt == 0:
            if dset_split == 0: 
                nrg_df = pd.read_csv(cwd+'/mt_results/'+dset_type+'/seed_'+str(seed) \
                        +'_'+labels_type \
                          +'_'+nn_style+'_nrg.csv')
            else:
                nrg_df = pd.read_csv(cwd+'/mt_results/'+split_type+'/seed_'+str(seed)\
                        +'_'+labels_type \
                          +'_'+nn_style+'_nrg.csv')
        else:
            if dset_split == 0:
                nrg_df = pd.read_csv(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e)+'_'\
                          +'seed_'+str(seed)+'_'+labels_type \
                          +'_'+nn_style+'_nrg.csv')       
                    
                nrg_df2 = pd.read_csv(cwd+'/mt_results/'+dset_type+'/NRG'+str(phi_e)\
                        +'_seed_'+str(seed)+'_st_det_'+labels_type \
                          +'_'+nn_style+'_nrg.csv')                          
            else:
                nrg_df = pd.read_csv(cwd+'/mt_results/'+split_type+'/NRG'+str(phi_e)+'_'\
                          +'seed_'+str(seed)+labels_type \
                          +'_'+nn_style+'_nrg.csv') 
                    
                nrg_df2 = pd.read_csv(cwd+'/mt_results/'+split_type+'/NRG'+str(phi_e)\
                        +'_seed_'+str(seed)+'_st_det_'+labels_type \
                          +'_'+nn_style+'_nrg.csv')                      
        # print(nrg_df.head(5))
        nrg_df.rename(columns={'rng':'rng_psi'})
        nrg_df2.rename(columns={'rng':'rng_alpha'})
        tdf = pd.concat([nrg_df,nrg_df2],axis=1)
        if seed == 1:
            nrg_per_dset = deepcopy(tdf)/len(seeds)
        else:
            nrg_per_dset += tdf/len(seeds)

    print('current dataset:'+dset_type)
    print(nrg_per_dset)
# %% present/record the data to be used in the table
# tdf = pd.DataFrame()
# tdf['Row_Labels'] = ['Accuracy','Energy']

# tdf['Ours_Max'] = [our_acc_max,our_nrg_max]
# tdf['Ours_Min'] = [our_acc_min,our_nrg_min]

# tdf['Alpha_Rng'] = [alpha_r_acc,alpha_r_nrg]
# tdf['ST_Rng'] = [sta_r_acc,sta_r_nrg]

# tdf['Alpha_H1'] = [alpha_h1_acc,alpha_h1_nrg]
# tdf['ST_H1'] = [sta_h1_acc,sta_h1_nrg]

# tdf['Alpha_H2'] = [alpha_h2_acc,alpha_h2_nrg]
# tdf['ST_H2'] = [sta_h2_acc,sta_h2_nrg]

# tdf.to_csv(cwd+'/mt_nrg_results/'+dset_type+'_NRG_df.csv')




