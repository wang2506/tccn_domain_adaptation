# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle as pk
import os
import numpy as np

# from optim_utils.optim_parser import optim_parser

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

# div imports
with open(cwd+'/ablation_study/unif_maxdiv_pairs','rb') as f:
    div_pairs = pk.load(f)/100

with open(cwd+'/ablation_study/div_pairs_extreme','rb') as f:
    div_pairs_e = pk.load(f)/100

with open(cwd+'/ablation_study/div_pairs_random','rb') as f:
    div_pairs_r = pk.load(f)/100

with open(cwd+'/ablation_study/div_pairs_estimated','rb') as f:
    div_pairs_est = pk.load(f)/100

# alpha imports
with open(cwd+'/ablation_study/alpha_vals_unif_maxdiv','rb') as f:
    tc_alpha = pk.load(f)

with open(cwd+'/ablation_study/alpha_vals_extreme','rb') as f:
    tc_alpha_div_e = pk.load(f)

with open(cwd+'/ablation_study/alpha_vals_random','rb') as f:
    tc_alpha_div_r = pk.load(f)

with open(cwd+'/ablation_study/alpha_vals_estimated','rb') as f:
    tc_alpha_div_est = pk.load(f)

# psi imports
with open(cwd+'/ablation_study/psi_vals_unif_maxdiv','rb') as f:
    psi_vals = pk.load(f)

with open(cwd+'/ablation_study/psi_vals_extreme','rb') as f:
    psi_vals_e = pk.load(f)

with open(cwd+'/ablation_study/psi_vals_random','rb') as f:
    psi_vals_r = pk.load(f)

with open(cwd+'/ablation_study/psi_vals_estimated','rb') as f:
    psi_vals_est = pk.load(f)

# %% processing the data
# psi to binary variable
psi_vals = [int(np.round(j,0)) for j in psi_vals[len(psi_vals.keys())-1]]
psi_vals_e = [int(np.round(j,0)) for j in psi_vals_e[len(psi_vals_e.keys())-1]]
psi_vals_r = [int(np.round(j,0)) for j in psi_vals_r[len(psi_vals_r.keys())-1]]
psi_vals_est = [int(np.round(j,0)) for j in psi_vals_est[len(psi_vals_est.keys())-1]]

# rescale alphas to sum to 1 [maintain static ratios]
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

rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = rescale_alphas(psi_vals,tc_alpha)

rc_se_alpha, rc_te_alpha,joint_alpha_e, s_inds_e, t_inds_e = rescale_alphas(psi_vals_e,tc_alpha_div_e)

rc_sr_alpha, rc_tr_alpha, joint_alpha_r, s_inds_r, t_inds_r  = rescale_alphas(psi_vals_r,tc_alpha_div_r)

rc_sest_alpha, tc_test_alpha, joint_alpha_est, s_inds_est, t_inds_est = \
    rescale_alphas(psi_vals_est,tc_alpha_div_est)

# %% ovr plot settings
gs_kw = dict(width_ratios=[1,1,1], height_ratios=[2,1,2])
fig,ax = plt.subplots(3,3,figsize=(8,5),\
        gridspec_kw=gs_kw,dpi=250)
plt.subplots_adjust(hspace=0.5)
# %% source/target classification plot [psi]
base_psi_x = list(range(len(psi_vals)))
cats = 3
width = 0.6
spacing = np.round(width/cats,2)+5e-2

ax[1,0].scatter([cval for cval in base_psi_x],psi_vals,marker='+',c='darkblue',\
           label='Uniform',s=45)#36) edgecolor='black', 
ax[1,1].scatter([cval for cval in base_psi_x],psi_vals_e,marker='x',c='forestgreen',\
           label='Extreme',s=36)#46) edgecolor='black', 
ax[1,2].scatter([cval for cval in base_psi_x],psi_vals_r,marker='3',c='magenta',\
           label='Random',s=56) #edgecolor='black', 

r0_labels = ['Uniform Divergence\n(A1)','Extreme Divergence\n(B1)','Random Divergence\n(C1)']
r1_labels = ['(A2)','(B2)','(C2)']
r2_labels = ['(A3)','(B3)','(C3)']
for i in range(3):
    for j in range(3):
        ax[i,j].set_xticks([0,2,4,6,8,10])
        
        ax[i,j].grid(True)
        ax[i,j].set_axisbelow(True)
    
    ax[2,i].set_xticklabels(['1','3','5','7','9'])
    ax[2,i].set_xlim([-1,10])
    
    ax[1,i].set_yticks([0,1])
    ax[1,i].set_yticklabels(['Source\n'+r'$\psi_i=0$','Target\n'+r'$\psi_i=1$'])
    ax[1,i].set_ylim(bottom=-0.2,top=1.2)    
    ax[1,i].set_xlim([-1,10])
    ax[1,i].set_xticklabels([])
    
    ax[0,i].set_title(r0_labels[i],fontsize=10.5)
    ax[1,i].set_title(r1_labels[i],fontsize=10.5)
    ax[2,i].set_title(r2_labels[i],fontsize=10.5)

ax[1,1].set_yticklabels([])
ax[1,2].set_yticklabels([])
ax[1,0].set_ylabel('Classification')

ax[2,0].set_xlabel('Device Number')
ax[2,1].set_xlabel('Device Number')
ax[2,2].set_xlabel('Device Number')

# %% plot div pairs
for i in range(10):
    div_pairs[i,i] = 0
    div_pairs_e[i,i] = 0
    div_pairs_r[i,i] = 0
ax[0,0].pcolormesh(div_pairs,cmap='inferno_r')
ax[0,1].pcolormesh(div_pairs_e,cmap='inferno_r')
im = ax[0,2].pcolormesh(div_pairs_r,cmap='inferno_r')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = plt.axes([0.92, 0.48, 0.02, 0.4]) 
fig.colorbar(im, cax=cax)

for i in range(3):
    ax[0,i].set_yticks(range(0,11,2))
ax[0,0].set_ylabel('Divergence Pairs\n\nDevice Number')

# %% alpha_sum plot [offloading breakdown graphic] 
spacing = 0.75
x = np.arange(0,10,1)

color_vec = ['mediumblue','forestgreen','maroon','magenta','orange',\
              'tab:cyan','tab:olive','tab:pink','tab:gray','tab:brown']

r_sum = np.zeros(joint_alpha.shape[0])
for j_ind,jar_col in enumerate(joint_alpha):
    if j_ind in s_inds:#:< 5:
        ax[2,0].bar(x,jar_col,width=spacing,\
                color=color_vec[j_ind],edgecolor='black',\
                    bottom=r_sum,label='D:'+str(j_ind+1))
    # print(jar_col)
    r_sum += jar_col
ax[2,0].set_ylim([0,1.05]) 
ax[2,0].set_yticks(np.arange(0,1.1,0.25))
ax[2,0].set_ylabel('Received\nWeights'+r' $\alpha_{i,j}$')

r_sum = np.zeros(joint_alpha.shape[0])
for j_ind,jar_col in enumerate(joint_alpha_e):
    if j_ind in s_inds:#:< 5:
        ax[2,1].bar(x,jar_col,width=spacing,\
                color=color_vec[j_ind],edgecolor='black',\
                    bottom=r_sum,label='From Device '+str(j_ind+1))
    r_sum += jar_col
ax[2,1].set_ylim([0,1.05]) 
ax[2,1].set_yticklabels([])

r_sum = np.zeros(joint_alpha_r.shape[0])
for j_ind,jar_col in enumerate(joint_alpha_r):
    if j_ind in s_inds_r:
        ax[2,2].bar(x,jar_col,width=spacing,\
                color=color_vec[j_ind],edgecolor='black',\
                    bottom=r_sum)
    r_sum += jar_col
ax[2,2].set_ylim([0,1.05]) 
ax[2,2].set_yticklabels([])

# ax[2,1].legend()
h,l = ax[2,0].get_legend_handles_labels()
kw = dict(ncol=1,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax[2,2].legend(h,l,bbox_to_anchor=(1,0,0.42,0.2),\
                        mode='expand',fontsize=10,**kw)
ax[2,2].add_artist(leg1)

# for i in range(3):
#     ax2[i].grid(True)
#     ax2[i].set_axisbelow(True)

# # fig2.savefig(cwd+'/ablation_plots/model_ratios.png',dpi=1000,bbox_inches='tight')
# # fig2.savefig(cwd+'/ablation_plots/model_ratios.pdf',dpi=1000,bbox_inches='tight')

fig.savefig(cwd+'/ablation_plots/3x3.png',dpi=1000,bbox_inches='tight')
fig.savefig(cwd+'/ablation_plots/3x3.pdf',dpi=1000,bbox_inches='tight')
