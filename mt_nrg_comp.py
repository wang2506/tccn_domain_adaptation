# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import pandas as pd
from copy import deepcopy

cwd = os.getcwd()
# %% settings
# labels_type = 'iid'
labels_type = 'mild'
dset_split = 0
# dset_split = 1
# dset_split = 2
split_type = None
# nn_style = 'MLP'
nn_style = 'CNN'
nrg_mt = 1
# phi_e = 1e1
phi_e = 1e0
grad_rv = False
# grad_rv = True
fl = True

nrg_mt = 1
phi_e = 1e0

pd.set_option('display.max_columns', None)
# %% extract data
# seeds = [1,2,3,4,5]
seeds = [3]
dsets = ['M','U','MM']

nrg_per_dset = pd.DataFrame()
cols = ['ours','rng']
nrg_per_dset['ours'] = [0,0]

if grad_rv == True:
    end2 = 'gr'
else:
    end2 = ''

if dset_split == 0:
    for idt,dset_type in enumerate(['M','U','MM']):
        for ids,seed in enumerate(seeds):
            if dset_type == 'MM':
                end = '_base_6'
            else:
                end = ''
            if fl == True:
                prefl = 'fl'
            else:
                prefl = ''
                
            if nrg_mt == 0:
                nrg_df = pd.read_csv(cwd+'/mt_results2/'+dset_type+'/seed_'+str(seed)+'_'\
                        +labels_type \
                          +'_'+nn_style+end+end2+'_nrg_rf.csv')
                # nrg_df2 = pd.read_csv(cwd+'/mt_results/'+dset_type+'/st_det_seed'+str(seed)+'_'\
                #         +labels_type \
                #           +'_'+nn_style+end+end2+'_nrg.csv')
            else:
                nrg_df = pd.read_csv(cwd+'/mt_results2/'+dset_type+'/NRG'+str(phi_e)+'_'\
                        +'seed_'+str(seed)+'_'+labels_type \
                          +'_'+nn_style+prefl+end+end2+'_nrg_rf.csv')
                nrg_df2 = pd.read_csv(cwd+'/mt_results2/'+dset_type+'/NRG'+str(phi_e)\
                        +'_seed_'+str(seed)+'_st_det_'+labels_type \
                          +'_'+nn_style+prefl+end+end2+'_nrg_rf.csv')

            nrg_df = nrg_df.rename(columns={'rng':'rng_alpha'})
            nrg_df2 = nrg_df2.rename(columns={'rng':'rng_psi'})
            tdf = pd.concat([nrg_df,nrg_df2],axis=1)
            
            if ids == 0:
                nrg_per_dset = deepcopy(tdf)/len(seeds)
            else:
                nrg_per_dset += tdf/len(seeds)

        print('current dataset:'+dset_type)
        print(nrg_per_dset/max(nrg_per_dset.loc[0,:]))
        print('\n')
else: #if dset_split == 1 :
    for split_type in ['M+MM','M+U','MM+U']:
        for ids,seed in enumerate(seeds):
            if dset_split == 1:
                pre = ''
            elif dset_split == 2:
                pre = 'total_'
            if 'MM' in split_type:
                end = '_base_6'
            else:
                end = ''
            
            if fl == True:
                prefl = 'fl'
            else:
                prefl = ''            
            
            if nrg_mt == 0:
                nrg_df = pd.read_csv(cwd+'/mt_results2/'+split_type+'/'+pre+'seed_'+str(seed)+'_'\
                        +labels_type \
                          +'_'+nn_style+end+end2+'_nrg_rf.csv')
                # nrg_df2 = pd.read_csv(cwd+'/mt_results/'+split_type+'/'+pre+'st_det_seed_'+str(seed)+'_'\
                #         +labels_type \
                #           +'_'+nn_style+end+end2+'_nrg.csv')
            else:
                nrg_df = pd.read_csv(cwd+'/mt_results2/'+split_type+'/NRG'+str(phi_e)+'_'\
                        +pre+'seed_'+str(seed)+'_'+labels_type \
                          +'_'+nn_style+prefl+end+end2+'_nrg_rf.csv')   
                nrg_df2 = pd.read_csv(cwd+'/mt_results2/'+split_type+'/NRG'+str(phi_e)+'_'\
                        +pre+'seed_'+str(seed)+'_st_det_'+labels_type \
                          +'_'+nn_style+prefl+end+end2+'_nrg_rf.csv')          

            nrg_df = nrg_df.rename(columns={'rng':'rng_alpha'})
            nrg_df2 = nrg_df2.rename(columns={'rng':'rng_psi'})
            tdf = pd.concat([nrg_df,nrg_df2],axis=1)
            
            if ids == 0:
                nrg_per_dset = deepcopy(tdf)/len(seeds)
            else:
                nrg_per_dset += tdf/len(seeds)

        print('current dataset:'+split_type)
        print(nrg_per_dset/max(nrg_per_dset.loc[0,:]))
        print('\n')

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




