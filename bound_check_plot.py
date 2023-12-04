# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:15:16 2023

@author: ch5b2
"""

import os
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

# %% init vars
all_dset_splits = [0,1,2]
all_dset_types = ['M','U','MM']
all_split_types = ['M+MM','M+U','MM+U']
phi_e = 1e0
labels_type = 'mild' #noniid
div_nn = 'CNN'

cwd = os.getcwd()

# %% data imports
for dset_split in all_dset_splits:
    if dset_split == 0: # only one dataset
        single_dat = {} 
        left = []
        right = []
        right2 = []
        for dset_type in all_dset_types:
            with open(cwd+'/bound_check2/'+dset_type+'_'+str(phi_e)+'_'\
                      +'_'+labels_type \
                      +'_'+div_nn+'_left','rb') as f:
                left_thm2_check = pk.load(f)
            with open(cwd+'/bound_check2/'+dset_type+'_'+str(phi_e)+'_'\
                      +'_'+labels_type \
                      +'_'+div_nn+'_right','rb') as f:
                right_thm2_check = pk.load(f)
            with open(cwd+'/bound_check2/'+dset_type+'_'+str(phi_e)+'_'\
                      +'_'+labels_type \
                      +'_'+div_nn+'_rightc','rb') as f:
                right_coro1_check = pk.load(f)   
            
            left.append(np.average(list(left_thm2_check.values())))
            right.append(np.average(list(right_thm2_check.values())))
            right2.append(np.average(list(right_coro1_check.values())))
        single_dat['left'] = np.round(left,2)
        single_dat['right'] = np.round(right,2)
        single_dat['right2'] = np.round(right2,2)
    else:
        if dset_split == 1:
            mixed_dat = {}   
        elif dset_split == 2:
            split_dat = {}
        l,r,r2 = [],[],[]            
        for split_type in all_split_types:
            if dset_split == 1:
                split_type = split_type
            elif dset_split == 2:
                split_type = split_type.replace('+','-')
            
            with open(cwd+'/bound_check2/'+split_type+'_'+str(phi_e)+'_'\
                      +'_'+labels_type \
                      +'_'+div_nn+'_left','rb') as f:
                left_thm2_check = pk.load(f)
            with open(cwd+'/bound_check2/'+split_type+'_'+str(phi_e)+'_'\
                      +'_'+labels_type \
                      +'_'+div_nn+'_right','rb') as f:
                right_thm2_check = pk.load(f)
            with open(cwd+'/bound_check2/'+split_type+'_'+str(phi_e)+'_'\
                      +'_'+labels_type \
                      +'_'+div_nn+'_rightc','rb') as f:
                right_coro1_check = pk.load(f)   
            
            l.append(np.average(list(left_thm2_check.values())))
            r.append(np.average(list(right_thm2_check.values())))
            r2.append(np.average(list(right_coro1_check.values())))
            
            if dset_split == 1:
                mixed_dat['left'] = np.round(l,2)
                mixed_dat['right'] = np.round(r,2)
                mixed_dat['right2'] = np.round(r2,2)
            elif dset_split == 2:
                split_dat['left'] = np.round(l,2)
                split_dat['right'] = np.round(r,2)
                split_dat['right2'] = np.round(r2,2)
            

# %% data gathering and stuff



# %% plots

