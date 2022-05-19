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
pwd = os.path.dirname(cwd)
# %% load data 
obj_f = '/obj_val/'
psi_f = '/psi_val/'
hat_ep_f = '/hat_ep_val/'
alpha_f = '/alpha_val/'

obj_vals = {}
psi_vals = {}
hat_ep_vals = {}
alpha_vals = {}

# phi_e = [1,10,1e3,1e4,1e5,1e6]
# phi_e = [1e0,1e1,1e2]
phi_e = [1e-2,1e0,1e2]  
tds = 10
tseed = 1
model = 'MLP'
# dataset = 'M'
dataset = 'U'
# dataset = 'MM'
iid = 'mild'

# if dataset == 'M' or dataset == 'U':
#     phi_e = [1e-2,1e0,1e2]    
# else:
#     phi_e = [1e-1,1e0,1e1]

# alpha imports
for tpe in phi_e:
    with open(pwd+'/optim_results/'+alpha_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid,'rb') as f:
        tc_alpha = pk.load(f)
    alpha_vals[tpe] = tc_alpha

# psi imports
for tpe in phi_e:
    with open(pwd+'/optim_results/'+psi_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid,'rb') as f:
        tc_psi = pk.load(f)
        
    tc_psi = [int(np.round(j,0)) for j in tc_psi[len(tc_psi.keys())-1]]
    psi_vals[tpe] = tc_psi

# input('a')
# %% rescale alphas to sum to 1 [maintain static ratios]
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
            # print(div_factor)
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

tj_alphas = {}
tt_alphas = {}
for tpe in phi_e:
    rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = rescale_alphas(psi_vals[tpe],alpha_vals[tpe])
    tj_alphas[tpe] = joint_alpha
    tt_alphas[tpe] = rc_t_alpha

# # have a small rounding issue, manually change
# tj_alphas[1e4][3,5] -= 0.01
# tj_alphas[1e4][3,6] -= 0.01
# tj_alphas[1e3][3,5] += 0.01

# tt_alphas[1e4][3,0] -= 0.01
# tt_alphas[1e4][3,1] -= 0.01
# tt_alphas[1e3][3,0] += 0.01

# input('a')
# %% plot bars for max-min
fig,ax = plt.subplots(3,1,figsize=(3,5),dpi=250,sharex=True) #vertically stacked bars
#plt.subplots(1,1,figsize=(5,2),dpi=250)#,sharey=True)
# cats = 3
cats = 1
width = 0.5
spacing = np.round(width/cats,2)
x = np.arange(sum(psi_vals[1e2])) # TODO special case - we know that 5 sources every time
# x = np.arange(0,10,1)

color_vec = ['mediumblue','forestgreen','maroon','magenta','orange']

# for itpe,tpe in enumerate(phi_e):
#     r_sum = np.zeros(joint_alpha.shape[0])
#     for j_ind,jar_col in enumerate(tj_alphas[tpe]):
#         if j_ind in s_inds:#:< 5:
#             ax.bar(x+(-1+itpe)*spacing,jar_col,width=spacing,\
#                    color=color_vec[j_ind],edgecolor='black',\
#                        bottom=r_sum,label='S'+str(j_ind+1))
#         r_sum += jar_col

for itpe,tpe in enumerate(phi_e):
    r_sum = np.zeros(rc_t_alpha.shape[1])
    for j_ind,jar_col in enumerate(tt_alphas[tpe]):
        if j_ind in s_inds:#:< 5:
            ax[itpe].bar(x,jar_col,width=spacing,\
                   color=color_vec[j_ind],edgecolor='black',\
                       bottom=r_sum,label='S'+str(j_ind+1))
        r_sum += jar_col
#+(-1+itpe)*spacing
ax[2].set_xlabel('Target Device')
ax[2].set_xticks(range(5))
ax[2].set_xticklabels(range(5,11))

for i in range(3):
    ax[i].grid(True)
    ax[i].set_axisbelow(True)

ax[0].set_ylabel(r'$\phi_e = 1e-2$')
ax[1].set_ylabel(r'$\phi_e = 1e0$')
ax[2].set_ylabel(r'$\phi_e = 1e2$')

h,l = ax[0].get_legend_handles_labels()
# l = l[:5]
# h = h[:5]
kw = dict(ncol=2,loc = 'lower center',frameon=False)
kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax[0].legend(h[:3],l[:3],bbox_to_anchor=(-0.1,1.12,1.15,0.2),\
                        mode='expand',fontsize=10,**kw2) #,**kw2)
leg2 = ax[0].legend(h[3:],l[3:],bbox_to_anchor=(-0.1,0.98,1.15,0.2),\
                        mode='expand',fontsize=10,**kw)
ax[0].add_artist(leg1)

fig.text(-0.15,0.35,r'Model Weights $\alpha_{i,j}$',rotation='vertical',fontsize=12)

# %% energy computations for large range of phi_e values
if dataset == 'M':
    phi_e = [1e-2,1e-1]+[1e0,3e0,5e0,7e0,9e0]+[1e1,1e2,1e3]#,1e4] #2e1,4e1,6e1,1e2]
    #1e-3,1e-2,
elif dataset == 'U':
    phi_e = [1e-3,1e-2,1e-1]+[1e0,1e1,2e1,3e1,4e1,5e1]+[1e2,1e3]#,1e3]#,1e4] #,6e1,7e1,9e1
elif dataset == 'MM':
    phi_e = [1e-1,1e0,1e1]+[1.2e1,1.4e1,1.6e1,1.8e1]+[2e1,1e2,1e3] #,3e1,4e1,5e1]+[
param_2_bits = 1e9

# load in the rate constants
with open(pwd+'/nrg_constants/devices'+str(tds)\
    +'_d2dtxrates','rb') as f:
    d2d_rates = pk.load(f)
    
with open(pwd+'/nrg_constants/devices'+str(tds)\
    +'_txpowers','rb') as f:
    tx_powers = pk.load(f)

# alpha imports
for tpe in phi_e:
    with open(pwd+'/optim_results/'+alpha_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid,'rb') as f:
        tc_alpha = pk.load(f)
    alpha_vals[tpe] = tc_alpha

# psi imports
for tpe in phi_e:
    with open(pwd+'/optim_results/'+psi_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid,'rb') as f:
        tc_psi = pk.load(f)
        
    tc_psi = [int(np.round(j,0)) for j in tc_psi[len(tc_psi.keys())-1]]
    psi_vals[tpe] = tc_psi

rd_alphas = {}
for tpe in phi_e:
    rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = rescale_alphas(psi_vals[tpe],alpha_vals[tpe])
    rd_alphas[tpe] = joint_alpha

## calculate energy use based on alphas
tpe_nrg = []
tpe_param_tx = []
for tpe in phi_e:
    c_nrg = 0
    p_tx = 0
    for i in range(tds): #for every device
        tc_alpha = rd_alphas[tpe][i,:]
        for ind_ca,ca in enumerate(tc_alpha):
            if ca > 1e-3: #1e-3
                c_nrg += param_2_bits/d2d_rates[i,ind_ca] * tx_powers[i] #* ca
                p_tx += 1
    tpe_nrg.append(c_nrg)
    tpe_param_tx.append(p_tx)

norm_tpe_nrg = np.round(np.array(tpe_nrg)/np.max(tpe_nrg),4)
if dataset == 'M':
    tpe_param_tx = 16-np.array(tpe_param_tx) #16 is typically max tx
elif dataset == 'U':
    tpe_param_tx = 16-np.array(tpe_param_tx) 
elif dataset == 'MM':
    tpe_param_tx = 20-np.array(tpe_param_tx)

# %% plot the normalized energy result
if dataset == 'M':
    grid_ratios = {'width_ratios': [0.6,3,1]} 
elif dataset == 'U':
    grid_ratios = {'width_ratios': [2,2.5,0.6]} 
elif dataset == 'MM':
    grid_ratios = {'width_ratios': [0.8,3,0.8]} 

fig2,ax2 = plt.subplots(1,3,figsize=(6,4),dpi=250,sharey=True,gridspec_kw=grid_ratios)
# fig2,ax2 = plt.subplots(1,1,figsize=(6,4),dpi=250)#,sharey=True,gridspec_kw=grid_ratios)
twin2_2 = ax2[2].twinx()
# twin2_2.set_ylabel('Saved Transmissions',fontsize=14)
# twin2_2.set_ylim([0,max(tpe_param_tx)+1])
twin2_2.get_yaxis().set_ticklabels([])
twin2_2.get_yaxis().set_ticks([])

ax2[0].tick_params(axis='y', colors='darkblue')
# twin2_2.tick_params(axis='y', colors='darkgreen')
ax2[0].yaxis.label.set_color('darkblue')
# twin2_2.yaxis.label.set_color('darkgreen')

if dataset == 'M':
    ax2[0].step(range(2),norm_tpe_nrg[:2],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_0 = ax2[0].twinx()
    twin2_0.step(range(2),tpe_param_tx[:2],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_0.set_ylim([-1,12])
    twin2_0.get_yaxis().set_ticklabels([])#.set_visible(False)  


    ax2[1].step(range(5),norm_tpe_nrg[2:7],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_1 = ax2[1].twinx()
    twin2_1.step(range(5),tpe_param_tx[2:7],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_1.set_ylim([-1,12])
    twin2_1.get_yaxis().set_ticklabels([])#.set_visible(False)    
    
    
    ax2[2].step(range(3),norm_tpe_nrg[7:],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_2 = ax2[2].twinx()
    twin2_2.step(range(3),tpe_param_tx[7:],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_2.set_ylim([-1,12])
    # ax2.set_xticks(range(9))
    # ax2.set_xticklabels(['1e-1','1e0','3e0','5e0','7e0','9e0','1e1','1e2']+\
    #         ['1e3'],fontsize=12)
    
    ax2[0].set_xticks(range(2))
    ax2[0].set_xticklabels(['1e-2','1e-1'])
    
    ax2[1].set_xticks(range(5))
    ax2[1].set_xticklabels(['1e0','3e0','5e0',\
                '7e0','9e0'])#,'2e1'])
    
    ax2[2].set_xticks(range(3))
    ax2[2].set_xticklabels(['1e1','1e2','1e3'])
    
    # twin2_2.set_yticklabels([0]+np.arange(0,12,2).tolist(),fontsize=12)    
    ax2[2].set_yticklabels(['40']+[str(i) for i in np.arange(40,101,10)],fontsize=12)
elif dataset == 'U':
    ax2[0].step(range(4),norm_tpe_nrg[:4],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_0 = ax2[0].twinx()
    twin2_0.step(range(4),tpe_param_tx[:4],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_0.set_ylim([-1,12])
    twin2_0.get_yaxis().set_ticklabels([])#.set_visible(False)  


    ax2[1].step(range(5),norm_tpe_nrg[4:9],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_1 = ax2[1].twinx()
    twin2_1.step(range(5),tpe_param_tx[4:9],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_1.set_ylim([-1,12])
    twin2_1.get_yaxis().set_ticklabels([])#.set_visible(False)    
    
    
    ax2[2].step(range(2),norm_tpe_nrg[9:],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_2 = ax2[2].twinx()
    twin2_2.step(range(2),tpe_param_tx[9:],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_2.set_ylim([-1,12])
    
    ax2[0].set_xticks(range(4))
    ax2[0].set_xticklabels(['1e-3','1e-2','1e-1','1e0'])
    
    ax2[1].set_xticks(range(5))
    ax2[1].set_xticklabels(['1e1','2e1','3e1',\
                '4e1','5e1'])#,'2e1'])
    
    ax2[2].set_xticks(range(2))
    ax2[2].set_xticklabels(['1e2','1e3'])    
    
    # twin2_2.set_yticklabels([0]+np.arange(0,12,2).tolist(),fontsize=12)         
    ax2[0].set_yticklabels(['40']+[str(i) for i in np.arange(40,101,10)],fontsize=12)
elif dataset == 'MM':
    ax2[0].step(range(2),norm_tpe_nrg[:2],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_0 = ax2[0].twinx()
    twin2_0.step(range(2),tpe_param_tx[:2],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_0.set_ylim([-1,15])
    twin2_0.get_yaxis().set_ticklabels([])#.set_visible(False)    
    
    ax2[1].step(range(6),norm_tpe_nrg[2:8],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_1 = ax2[1].twinx()
    twin2_1.step(range(6),tpe_param_tx[2:8],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_1.set_ylim([-1,15])
    twin2_1.get_yaxis().set_ticklabels([])#.set_visible(False)    

    ax2[2].step(range(2),norm_tpe_nrg[8:],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_2 = ax2[2].twinx()
    twin2_2.step(range(2),tpe_param_tx[8:],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_2.set_ylim([-1,15])
    # twin2_2.get_yaxis().set_ticklabels([])#.set_visible(False)        

    ax2[0].set_xticks(range(2))
    ax2[0].set_xticklabels(['1e-1','1e0'])
    
    ax2[1].set_xticks(range(6))
    ax2[1].set_xticklabels(['1e1','1.2e1','1.4e1',\
                '1.6e1','1.8e1','2e1'])
    
    ax2[2].set_xticks(range(2))
    ax2[2].set_xticklabels(['1e2','1e3'])
    
    # ax2.set_xticklabels(['1e0','1e1','1.2e1','1.4e1',\
    #             '1.6e1','1.8e1','2e1','1e2','1e3'],fontsize=12)
    
    # twin2_2.set_yticklabels([0]+np.arange(0,16,2).tolist(),fontsize=12)        
    ax2[0].set_yticklabels(['30']+[str(i) for i in np.arange(30,101,10)],fontsize=12)

twin2_2.set_ylabel('Saved Transmissions',fontsize=14)
twin2_2.tick_params(axis='y', colors='darkgreen')
twin2_2.yaxis.label.set_color('darkgreen')

# ax2.grid(True)
for i in range(3):
    ax2[i].grid(True)

ax2[0].set_ylabel('Normalized Energy \n Consumption (%)',fontsize=14)
# ax2[0].set_yticks([])

fig2.text(0.5, 0, r'$\phi_e$', ha='center',fontsize=14)
# fig2.tight_layout()
fig2.subplots_adjust(wspace=0.2)#25)

# %% save figures
# fig.savefig(cwd+'/nrg_plots/nrg_model_ratios'+dataset+'.png',dpi=1000,bbox_inches='tight')
# fig.savefig(cwd+'/nrg_plots/nrg_model_ratios'+dataset+'.pdf',dpi=1000,bbox_inches='tight')

# fig2.savefig(cwd+'/nrg_plots/nrg_norm_nrg'+dataset+'.png',dpi=1000,bbox_inches='tight')
# fig2.savefig(cwd+'/nrg_plots/nrg_norm_nrg'+dataset+'.pdf',dpi=1000,bbox_inches='tight')

# %% old
# if dataset == 'M':
#     grid_ratios = {'width_ratios': [0.6,3,0.6]}
# elif dataset == 'U':
#     grid_ratios = {'width_ratios': [1,3,0.6]}
# elif dataset == 'MM':
#     grid_ratios = {}

    # ax2[0].step(range(2),norm_tpe_nrg[:2],where='post', \
    #         marker='x',linestyle='dashed',color='darkblue')
    # ax2[1].step(range(8),norm_tpe_nrg[2:10],where='post', \
    #         marker='x',linestyle='dashed',color='darkblue')
    # ax2[2].step(range(2),norm_tpe_nrg[10:],where='post', \
    #         marker='x',linestyle='dashed',color='darkblue')

    # twin2_0 = ax2[0].twinx()
    # twin2_0.step(range(2),tpe_param_tx[:2],where='post', \
    #         marker='x',linestyle='dashed',color='darkgreen')
    # twin2_0.set_ylim([-1,10])
    # twin2_0.get_yaxis().set_ticklabels([])#.set_visible(False)
    
    # twin2_1 = ax2[1].twinx()
    # twin2_1.step(range(8),tpe_param_tx[2:10],where='post', \
    #         marker='x',linestyle='dashed',color='darkgreen')
    # twin2_1.set_ylim([-1,10])    
    # twin2_1.get_yaxis().set_ticklabels([])
    
    # twin2_2.step(range(2),tpe_param_tx[10:],where='post', \
    #         marker='x',linestyle='dashed',color='darkgreen')    
    
    # ax2[0].set_xticks(range(2))
    # ax2[0].set_xticklabels(['1e-3','1e-2'])#'1e0'])
    
    # ax2[1].set_xticks(range(8)) 
    # ax2[1].set_xticklabels(['1e-1','1e0','3e0','5e0','7e0','9e0','1e1','1e2'])#,'2.2e4'])
    
    # ax2[2].set_xticks(range(2))
    # ax2[2].set_xticklabels(['1e3','1e4'])    
    
    
# elif dataset == 'U':
#     ax2[0].step(range(3),norm_tpe_nrg[:3],where='post', \
#             marker='x',linestyle='dashed',color='darkblue')
#     ax2[1].step(range(6),norm_tpe_nrg[3:9],where='post', \
#             marker='x',linestyle='dashed',color='darkblue')
#     ax2[2].step(range(2),norm_tpe_nrg[9:],where='post', \
#             marker='x',linestyle='dashed',color='darkblue')

#     twin2_0 = ax2[0].twinx()
#     twin2_0.step(range(3),tpe_param_tx[:3],where='post', \
#             marker='x',linestyle='dashed',color='darkgreen')
#     twin2_0.set_ylim([-1,10])
#     twin2_0.get_yaxis().set_ticklabels([])#.set_visible(False)
    
#     twin2_1 = ax2[1].twinx()
#     twin2_1.step(range(6),tpe_param_tx[3:9],where='post', \
#             marker='x',linestyle='dashed',color='darkgreen')
#     twin2_1.set_ylim([-1,10])    
#     twin2_1.get_yaxis().set_ticklabels([])
    
#     twin2_2.step(range(2),tpe_param_tx[9:],where='post', \
#             marker='x',linestyle='dashed',color='darkgreen')    

#     ax2[0].set_xticks(range(3))
#     ax2[0].set_xticklabels(['1e-3','1e-2','1e-1'])#'1e0'])
    
#     ax2[1].set_xticks(range(6)) 
#     ax2[1].set_xticklabels(['1e0','1e1','2e1','3e1','4e1','5e1'])#,'2.2e4'])
    
#     ax2[2].set_xticks(range(2))
#     ax2[2].set_xticklabels(['1e2','1e3'])        
    