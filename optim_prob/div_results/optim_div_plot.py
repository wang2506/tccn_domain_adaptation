# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:46:45 2022

@author: henry
"""
import matplotlib.pyplot as plt
import pickle as pk
import os
import numpy as np

# from optim_utils.optim_parser import optim_parser

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

rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = rescale_alphas(psi_vals,tc_alpha)
# joint_alpha = np.concatenate((rc_s_alpha,rc_t_alpha),axis=1)

rc_se_alpha, rc_te_alpha,joint_alpha_e, s_inds_e, t_inds_e = rescale_alphas(psi_vals_e,tc_alpha_div_e)
# joint_alpha_e = np.concatenate((rc_se_alpha,rc_te_alpha),axis=1)

rc_sr_alpha, rc_tr_alpha, joint_alpha_r, s_inds_r, t_inds_r  = rescale_alphas(psi_vals_r,tc_alpha_div_r)
# joint_alpha_r = np.concatenate((rc_sr_alpha,rc_tr_alpha),axis=1)

rc_sest_alpha, tc_test_alpha, joint_alpha_est, s_inds_est, t_inds_est = \
    rescale_alphas(psi_vals_est,tc_alpha_div_est)

# %% source/target classification plot [psi]
fig,ax = plt.subplots(figsize=(5,1.5),dpi=250)

base_psi_x = list(range(len(psi_vals)))
# cats = 4 #3
# width = 0.8 #0.6 #0.84
cats = 3
width = 0.6
spacing = np.round(width/cats,2)+5e-2

ax.scatter([cval-1*spacing for cval in base_psi_x],psi_vals,marker='+',c='darkblue',\
           label='Uniform',s=45)#36) edgecolor='black', 
ax.scatter([cval-0*spacing for cval in base_psi_x],psi_vals_e,marker='x',c='forestgreen',\
           label='Extreme',s=36)#46) edgecolor='black', 
ax.scatter([cval+1*spacing for cval in base_psi_x],psi_vals_r,marker='3',c='magenta',\
           label='Random',s=56) #edgecolor='black', 
# ax.scatter([cval+1.5*spacing for cval in base_psi_x],psi_vals_est,marker='4',c='magenta',\
#            label='Estimated',s=56) #edgecolor='black', 

ax.set_xlabel('Device Number')
# ax.set_ylabel('Source/Target Classification')
# ax.set_title('Source/Target Classification')

ax.set_xticks(range(10))
ax.set_xticklabels(range(1,11))

ax.set_yticks([0,1])
ax.set_yticklabels(['Source','Target'])

ax.set_ylim(bottom=-0.2,top=1.2)

# ax.legend(ncol=3,loc='center',bbox_to_anchor=(0,0.38,1,0.2))
h,l = ax.get_legend_handles_labels()
kw = dict(ncol=3,loc = 'lower center',frameon=False)
# kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax.legend(h,l,bbox_to_anchor=(0,1,1,0.2),\
                        mode='expand',fontsize=10,**kw) #,**kw2)
# leg2 = ax.legend(h[3::1],l[3::1],bbox_to_anchor=(-0.25,1.02,2.4,0.2),\
                        # mode='expand',fontsize=10,**kw)
ax.grid(True)
ax.set_axisbelow(True)

# fig.savefig(cwd+'/ablation_plots/device_classification.png',dpi=1000,bbox_inches='tight')
# fig.savefig(cwd+'/ablation_plots/device_classification.eps',dpi=1000,bbox_inches='tight')

# %% alpha_sum plot [offloading breakdown graphic]
fig2,ax2 = plt.subplots(3,1,figsize=(3,5),dpi=250,sharex=True) #vertically stacked bars
width = 0.6 #0.69
cats = 1#3 #each bar is a target
spacing = np.round(width/cats,2)

x = np.arange(0,10,1)

color_vec = ['mediumblue','forestgreen','maroon','magenta','orange',\
             'tab:cyan','tab:olive','tab:pink','tab:gray','tab:brown']

r_sum = np.zeros(joint_alpha.shape[0])
for j_ind,jar_col in enumerate(joint_alpha):
    if j_ind in s_inds:#:< 5:
        ax2[0].bar(x-0*spacing,jar_col,width=spacing,\
               color=color_vec[j_ind],edgecolor='black',\
                   bottom=r_sum,label='S'+str(j_ind+1))
    # print(jar_col)
    r_sum += jar_col

r_sum = np.zeros(joint_alpha.shape[0])
for j_ind,jar_col in enumerate(joint_alpha_e):
    if j_ind in s_inds:#:< 5:
        ax2[1].bar(x+0*spacing,jar_col,width=spacing,\
               color=color_vec[j_ind],edgecolor='black',\
                   bottom=r_sum)#,label='Source '+str(j_ind+1))
    # print(jar_col)
    r_sum += jar_col

r_sum = np.zeros(joint_alpha_r.shape[0])
for j_ind,jar_col in enumerate(joint_alpha_r):
    if j_ind in s_inds_r:
        ax2[2].bar(x+0*spacing,jar_col,width=spacing,\
               color=color_vec[j_ind],edgecolor='black',\
                   bottom=r_sum)#,label='Source '+str(j_ind+1))
    # print(jar_col)
    r_sum += jar_col

# ax2[0].set_xlabel('Target Device')
fig2.text(0.5, 0.05, 'Target Device',ha='center',fontsize=12)
ax2[0].set_xticks(range(10))
ax2[0].set_xticklabels(range(1,11))

ax2[0].set_xlim([-0.2,9.5])
for i in range(3):
    ax2[i].set_ylim([0,1.05])
# ax2[1].set_ylim

# ax2.set_ylabel(r'Model Weights at Targets $\alpha_{i,j}$')

ax2[0].set_ylabel(r'Uniform')
ax2[1].set_ylabel(r'Extreme')
ax2[2].set_ylabel(r'Random')

fig2.text(-0.15,0.35,r'Model Weights $\alpha_{i,j}$',rotation='vertical',fontsize=12)

# ax2.set_title('Effect of Divergence Estimates on Alpha Value')

# ax2.legend()
h,l = ax2[0].get_legend_handles_labels()
kw = dict(ncol=2,loc = 'lower center',frameon=False)
kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax2[0].legend(h[:3],l[:3],bbox_to_anchor=(-0.1,1.12,1.15,0.2),\
                        mode='expand',fontsize=10,**kw2) #,**kw2)
leg2 = ax2[0].legend(h[3:],l[3:],bbox_to_anchor=(-0.1,0.98,1.15,0.2),\
                        mode='expand',fontsize=10,**kw)
ax2[0].add_artist(leg1)

for i in range(3):
    ax2[i].grid(True)
    ax2[i].set_axisbelow(True)

# fig2.savefig(cwd+'/ablation_plots/model_ratios.png',dpi=1000,bbox_inches='tight')
# fig2.savefig(cwd+'/ablation_plots/model_ratios.eps',dpi=1000,bbox_inches='tight')


# %% backups
# # tp_alpha = tc_alpha[:5,5:]
# sources = np.where(np.array(psi_vals[1])==0)
# targets = np.where(np.array(psi_vals[1])==1)

# tp_alpha = tc_alpha[sources][:,targets]
# tp_alpha2 = [ta[0].tolist() for ta in tp_alpha]
# for tai,taj in enumerate(tp_alpha2):
#     for ttai,ttaj in enumerate(taj):
#         taj[ttai] = np.round(ttaj,2)

# tc_alpha_div = tc_alpha_div[sources][:,targets]
# tc_alpha_div2 = [ta[0].tolist() for ta in tc_alpha_div]
# for tai,taj in enumerate(tc_alpha_div2):
#     for ttai,ttaj in enumerate(taj):
#         taj[ttai] = np.round(ttaj,2)





