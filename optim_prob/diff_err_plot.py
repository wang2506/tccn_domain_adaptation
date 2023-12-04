# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 19:44:38 2023

@author: ch5b2
"""
import os
import cvxpy as cp
import numpy as np
import pickle as pk
import random
from copy import deepcopy

cwd = os.getcwd()

# %% load in data
phi_e = 1e0
seed = 1
t_devices = 10
div_nn = 'MLP'
all_dset_type = ['M','U','MM']
labels_type = 'mild'
prefl = 'fl'
end2 = ''
pre_str = 'R3'

sav_dict_keys = ['obj_val','psi_val','hat_ep_val','alpha_val']

m_dict = {}
u_dict = {}
mm_dict = {}

nm_dict,nu_dict,nmm_dict = {},{},{}

for dset_type in all_dset_type:
    if dset_type == 'MM':
        end = '_base_6'
    else:
        end = ''
    
    for sav_key in sav_dict_keys:
        with open(cwd+'/optim_results/'+sav_key+'/'+pre_str+'NRG_'+str(phi_e)+'_'\
            +'devices'+str(t_devices)+'_seed'+str(seed)\
            +'_'+div_nn+'_'+dset_type+'_'+labels_type\
            +prefl+end+end2,'rb') as f:
            temp = pk.load(f)

        if dset_type == 'M':
            nm_dict[sav_key] = temp
        elif dset_type == 'U':
            nu_dict[sav_key] = temp
        elif dset_type == 'MM':
            nmm_dict[sav_key] = temp

    for sav_key in sav_dict_keys:
        with open(cwd+'/optim_results/'+sav_key+'/NRG_'+str(phi_e)+'_'\
            +'devices'+str(t_devices)+'_seed'+str(seed)\
            +'_'+div_nn+'_'+dset_type+'_'+labels_type\
            +''+''+end2,'rb') as f:
            temp = pk.load(f)

        if dset_type == 'M':
            m_dict[sav_key] = temp
        elif dset_type == 'U':
            u_dict[sav_key] = temp
        elif dset_type == 'MM':
            mm_dict[sav_key] = temp

# %% get the model accuracy changes
## no need, the model offloading ratios barely change

# %% 
def rescale_alphas(c_psi,c_alpha):
    # sources received models (alpha) adjust
    s_pv = np.where(np.array(c_psi) == 0)[0]
    s_alpha = c_alpha[:,s_pv]
    s_alpha[np.where(s_alpha <= 5e-2)] = 0
    
    s_alpha_sums = np.sum(s_alpha,axis=0)
    for div_factor in s_alpha_sums:
        if div_factor > 0:
            raise ValueError('Should not have greater than 0')
    
    # targets received models (alpha) adjust
    t_pv = np.where(np.array(c_psi) == 1)[0]
    t_alpha = c_alpha[:,t_pv]
    t_alpha[np.where(t_alpha <= 1e-2)] = 0 
    
    t_alpha_sums = np.sum(t_alpha,axis=0)
    for ind_f,div_factor in enumerate(t_alpha_sums):
        if div_factor > 0:
            t_alpha[:,ind_f] = [np.round(c_val/div_factor,2) for c_val in t_alpha[:,ind_f]]

    # concatenate the two alpha matrices based on numerical order
    for a_ind in range(c_alpha.shape[0]):
        if a_ind == 0:
            if a_ind in s_pv:
                ovr_alpha = s_alpha[:,np.where(s_pv==a_ind)[0]]
            else:
                ovr_alpha = t_alpha[:,np.where(t_pv==a_ind)[0]]
        else:
            if a_ind in s_pv:
                ovr_alpha = np.hstack((ovr_alpha,s_alpha[:,np.where(s_pv==a_ind)[0]]))
            else:
                ovr_alpha = np.hstack((ovr_alpha,t_alpha[:,np.where(t_pv==a_ind)[0]]))
        
    return s_alpha,t_alpha,ovr_alpha,s_pv,t_pv

# %% determine psi_vals from rounding
m_dict['psi_val'] = [int(np.round(j,0)) for j in m_dict['psi_val'][len(m_dict['psi_val'].keys())-1]]
u_dict['psi_val'] = [int(np.round(j,0)) for j in u_dict['psi_val'][len(u_dict['psi_val'].keys())-1]]
mm_dict['psi_val'] = [int(np.round(j,0)) for j in mm_dict['psi_val'][len(mm_dict['psi_val'].keys())-1]]

nm_dict['psi_val'] = [int(np.round(j,0)) for j in nm_dict['psi_val'][len(nm_dict['psi_val'].keys())-1]]
nu_dict['psi_val'] = [int(np.round(j,0)) for j in nu_dict['psi_val'][len(nu_dict['psi_val'].keys())-1]]
nmm_dict['psi_val'] = [int(np.round(j,0)) for j in nmm_dict['psi_val'][len(nmm_dict['psi_val'].keys())-1]]

# %% 
tj_alphas,tt_alphas = {},{}
ntj_alphas,ntt_alphas = {},{}

for dset_type in all_dset_type:
    if dset_type == 'M':
        rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = \
            rescale_alphas(m_dict['psi_val'],m_dict['alpha_val'])
        nrc_s_alpha, nrc_t_alpha, njoint_alpha, ns_inds, nt_inds = \
            rescale_alphas(nm_dict['psi_val'],nm_dict['alpha_val'])        
    elif dset_type == 'U':
        rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = \
            rescale_alphas(u_dict['psi_val'],u_dict['alpha_val'])
        nrc_s_alpha, nrc_t_alpha, njoint_alpha, ns_inds, nt_inds = \
            rescale_alphas(nu_dict['psi_val'],nu_dict['alpha_val'])
    elif dset_type == 'MM':
        rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = \
            rescale_alphas(mm_dict['psi_val'],mm_dict['alpha_val'])
        nrc_s_alpha, nrc_t_alpha, njoint_alpha, ns_inds, nt_inds = \
            rescale_alphas(nmm_dict['psi_val'],nmm_dict['alpha_val']) 
            
    tj_alphas[dset_type] = joint_alpha
    tt_alphas[dset_type] = rc_t_alpha
    
    ntj_alphas[dset_type] = njoint_alpha
    ntt_alphas[dset_type] = nrc_t_alpha

# %% plot the changes in alpha ratios
import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,1,figsize=(3,5),dpi=250,sharex=True) #vertically stacked bars
cats = 1
width = 0.5
spacing = np.round(width/cats,2)
x = np.arange(5) 

color_vec = ['mediumblue','forestgreen','maroon','magenta','orange']

for itpe,tpe in enumerate(all_dset_type):
    r_sum = np.zeros(rc_t_alpha.shape[1])
    for j_ind,jar_col in enumerate(ntt_alphas[tpe]):
        if j_ind in s_inds:#:< 5:
            ax[itpe].bar(x,jar_col,width=spacing,\
                   color=color_vec[j_ind],edgecolor='black',\
                       bottom=r_sum,label='S'+str(j_ind+1))
        r_sum += jar_col





