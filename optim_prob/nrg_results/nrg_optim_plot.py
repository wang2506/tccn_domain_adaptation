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
phi_e = [1e3,1e4,1e5]
tds = 10
tseed = 1
model = 'MLP'
dataset = 'M'
iid = 'mild'

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
    s_alpha[np.where(s_alpha <= 1e-2)] = 0
    
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

# have a small rounding issue, manually change
tj_alphas[1e4][3,5] -= 0.01
tj_alphas[1e4][3,6] -= 0.01
tj_alphas[1e3][3,5] += 0.01

tt_alphas[1e4][3,0] -= 0.01
tt_alphas[1e4][3,1] -= 0.01
tt_alphas[1e3][3,0] += 0.01

# input('a')
# %% plot bars for max-min
fig,ax = plt.subplots(3,1,figsize=(3,5),dpi=250,sharex=True) #vertically stacked bars
#plt.subplots(1,1,figsize=(5,2),dpi=250)#,sharey=True)
# cats = 3
cats = 1
width = 0.5
spacing = np.round(width/cats,2)
x = np.arange(sum(psi_vals[1e3])) # TODO special case - we know that 5 sources every time
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

ax[0].set_ylabel(r'$\phi_e = 1e3$')
ax[1].set_ylabel(r'$\phi_e = 1e4$')
ax[2].set_ylabel(r'$\phi_e = 1e5$')

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

# %% save figures
# fig.savefig(cwd+'/nrg_plots/nrg_model_ratios.png',dpi=1000,bbox_inches='tight')
# fig.savefig(cwd+'/nrg_plots/nrg_model_ratios.pdf',dpi=1000,bbox_inches='tight')


