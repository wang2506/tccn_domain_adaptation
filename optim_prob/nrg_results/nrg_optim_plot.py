# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pk
import pandas as pd
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

phi_e = [1e-2,1e0,1e2]
# phi_e = [1e0,1e2]
tds = 10
tseed = 10
# tseed = 3 #for mnist
# model = 'MLP'
model = 'CNN'
dataset = 'M'
# dataset = 'U'
# dataset = 'MM'
# iid = 'mild'
iid = 'iid' #very light non-iid

if dataset == 'MM':
    end = '_base_6'
else:
    end = ''
# alpha imports
for tpe in phi_e:
    with open(pwd+'/optim_results2/'+alpha_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid+'fl'+end,'rb') as f:
        tc_alpha = pk.load(f)
    alpha_vals[tpe] = tc_alpha

# psi imports
for tpe in phi_e:
    with open(pwd+'/optim_results2/'+psi_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid+'fl'+end,'rb') as f:
        tc_psi = pk.load(f)
        
    tc_psi = [int(np.round(j,0)) for j in tc_psi[len(tc_psi.keys())-1]]
    psi_vals[tpe] = tc_psi

# %% rescale alphas to sum to 1 [maintain static ratios]
def rescale_alphas(c_psi,c_alpha):
    # sources received models (alpha) adjust
    s_pv = np.where(np.array(c_psi) == 0)[0]
    s_alpha = c_alpha[:,s_pv]
    # s_alpha[np.where(s_alpha <= 3.6e-2)] = 0
    s_alpha = np.zeros_like(s_alpha)
    
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

tj_alphas = {}
tt_alphas = {}
for tpe in phi_e:
    rc_s_alpha, rc_t_alpha, joint_alpha, s_inds, t_inds = rescale_alphas(psi_vals[tpe],alpha_vals[tpe])
    tj_alphas[tpe] = joint_alpha
    tt_alphas[tpe] = rc_t_alpha

# %% plot bars for max-min
fig,ax = plt.subplots(3,1,figsize=(3,5),dpi=250,sharex=True) #vertically stacked bars
cats = 1
width = 0.5
spacing = np.round(width/cats,2)
x = np.arange(sum(psi_vals[1e2])) 

color_vec = ['mediumblue','forestgreen','maroon','magenta','orange']

for itpe,tpe in enumerate(phi_e):
    r_sum = np.zeros(rc_t_alpha.shape[1])
    for j_ind,jar_col in enumerate(tt_alphas[tpe]):
        if j_ind in s_inds:#:< 5:
            ax[itpe].bar(x,jar_col,width=spacing,\
                    color=color_vec[j_ind],edgecolor='black',\
                        bottom=r_sum,label='S'+str(j_ind+1))
        r_sum += jar_col

ax[2].set_xlabel('Target Device')
ax[2].set_xticks(range(5))
ax[2].set_xticklabels(range(5,10))

for i in range(3):
    ax[i].grid(True)
    ax[i].set_axisbelow(True)

ax[0].set_ylabel('Low Link Costs\n'+r'$\phi^{E} = 0.01$') #1e-2
ax[1].set_ylabel('Normal Link Costs\n'+r'$\phi^{E} = 1$') #1e0
ax[2].set_ylabel('High Link Costs\n'+r'$\phi^{E} = 100$') #1e2

h,l = ax[0].get_legend_handles_labels()
kw = dict(ncol=2,loc = 'lower center',frameon=False)
kw2 = dict(ncol=3,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax[0].legend(h[:3],l[:3],bbox_to_anchor=(-0.1,1.12,1.15,0.2),\
                        mode='expand',fontsize=10,**kw2) #,**kw2)
leg2 = ax[0].legend(h[3:],l[3:],bbox_to_anchor=(-0.1,0.98,1.15,0.2),\
                        mode='expand',fontsize=10,**kw)
ax[0].add_artist(leg1)

#-0.15,0.35
fig.text(-0.2,0.35,r'Model Weights $\alpha_{i,j}$',rotation='vertical',fontsize=12)
# fig.text(-0.22,0.25,r'Model Weights ($\alpha_{i,j}$) for ' \
#          +'\nThree Com Nrg Scales ($\phi^{E}$)',rotation='vertical',fontsize=12)

# ## save figures
# fig.savefig(cwd+'/nrg_plots/2nrg_model_ratios'+dataset+'.png',dpi=1000,bbox_inches='tight')
# fig.savefig(cwd+'/nrg_plots/2nrg_model_ratios'+dataset+'.pdf',dpi=1000,bbox_inches='tight')

# %% energy computations for large range of phi_e values
if dataset == 'M': ##  
    # phi_e = [1e-2,1e-1]+[1e0,3e0,5e0,1e1]+[1e2,1e3,1e4] #1e4 is new
    phi_e = [1e-3,1e-2,1e-1]+[1e0,3e0,5e0]+[1e1,1e2,1e3,1e4] #1e4 is new
elif dataset == 'U': ## 
    # phi_e = [1e-3,1e-2,1e-1]+[1e0,1e1,2e1,3e1,4e1,5e1]+[1e2,1e3]
     phi_e = [1e-2,1e-1]+[1e0,4e0,8e0,1e1]+[1e2,1e3,1e4]
elif dataset == 'MM': # 5e-1 + 1e4
    # phi_e = [1e-1,1e0,1e1]+[1.2e1,1.4e1,1.6e1,1.8e1]+[2e1,1e2,1e3]
    phi_e = [1e-3,1e-2]+[1e-1,5e-1,1e0,1e1,2e1]+[1e2,1e3,1e4]
param_2_bits = 1e9

# load in the rate constants
with open(pwd+'/nrg_constants/devices'+str(tds)\
    +'_d2dtxrates','rb') as f:
    d2d_rates = pk.load(f)
    
with open(pwd+'/nrg_constants/devices'+str(tds)\
    +'_txpowers','rb') as f:
    tx_powers = pk.load(f)

fl_flag = 'fl'
# alpha imports
for tpe in phi_e:
    with open(pwd+'/optim_results2/'+alpha_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid+fl_flag+end,'rb') as f:
        tc_alpha = pk.load(f)
    alpha_vals[tpe] = tc_alpha

# psi imports
for tpe in phi_e:
    with open(pwd+'/optim_results2/'+psi_f+'/NRG_'+str(tpe)\
        +'_devices'+str(tds)+'_seed'+str(tseed)+'_'\
        +model+'_'+dataset+'_'+iid+fl_flag+end,'rb') as f:
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
            if ca > 1e-3: 
                c_nrg += param_2_bits/d2d_rates[i,ind_ca] * tx_powers[i] 
                p_tx += 1
    tpe_nrg.append(c_nrg)
    tpe_param_tx.append(p_tx)

norm_tpe_nrg = np.round(np.array(tpe_nrg)/np.max(tpe_nrg),4)
if dataset == 'M':
    tpe_param_tx = np.max(tpe_param_tx)-np.array(tpe_param_tx) 
elif dataset == 'U':
    tpe_param_tx = np.max(tpe_param_tx)-np.array(tpe_param_tx) 
elif dataset == 'MM':
    tpe_param_tx = 20-np.array(tpe_param_tx)

# %% plot the normalized energy result
if dataset == 'M':
    grid_ratios = {'width_ratios': [0.6,3,1]} 
elif dataset == 'U':
    grid_ratios = {'width_ratios': [0.6,2.5,1.5]} 
elif dataset == 'MM':
    grid_ratios = {'width_ratios': [0.8,3,0.8]} 

fig2,ax2 = plt.subplots(1,3,figsize=(6,4),dpi=250,sharey=True,gridspec_kw=grid_ratios)
twin2_2 = ax2[2].twinx()
twin2_2.get_yaxis().set_ticklabels([])
twin2_2.get_yaxis().set_ticks([])

ax2[0].tick_params(axis='y', colors='darkblue')
ax2[0].yaxis.label.set_color('darkblue')

if dataset == 'M':
    ax2[0].step(range(2),norm_tpe_nrg[:2],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_0 = ax2[0].twinx()
    twin2_0.step(range(2),tpe_param_tx[:2],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_0.set_ylim([-1,14])
    twin2_0.get_yaxis().set_ticklabels([])

    ax2[1].step(range(5),norm_tpe_nrg[2:7],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_1 = ax2[1].twinx()
    twin2_1.step(range(5),tpe_param_tx[2:7],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_1.set_ylim([-1,14])
    twin2_1.get_yaxis().set_ticklabels([])
    
    ax2[2].step(range(3),norm_tpe_nrg[7:],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_2 = ax2[2].twinx()
    twin2_2.step(range(3),tpe_param_tx[7:],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_2.set_ylim([-1,14])

    ax2[0].set_xticks(range(2))
    ax2[0].set_xticklabels(['1e-3','1e-2'])
    
    ax2[1].set_xticks(range(5))
    ax2[1].set_xticklabels(['1e-1','1e0','3e0','5e0','1e1'])
    
    ax2[2].set_xticks(range(3))
    ax2[2].set_xticklabels(['1e2','1e3','1e4'])
    
    ax2[2].set_yticklabels(['0']+[str(i) for i in np.arange(0,101,20)],fontsize=12)
elif dataset == 'U':
    ax2[0].step(range(2),norm_tpe_nrg[:2],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_0 = ax2[0].twinx()
    twin2_0.step(range(2),tpe_param_tx[:2],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_0.set_ylim([-1,17])
    twin2_0.get_yaxis().set_ticklabels([]) 


    ax2[1].step(range(4),norm_tpe_nrg[2:6],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_1 = ax2[1].twinx()
    twin2_1.step(range(4),tpe_param_tx[2:6],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_1.set_ylim([-1,17])
    twin2_1.get_yaxis().set_ticklabels([]) 
    
    
    ax2[2].step(range(3),norm_tpe_nrg[6:],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_2 = ax2[2].twinx()
    twin2_2.step(range(3),tpe_param_tx[6:],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_2.set_ylim([-1,17])
    
    ax2[0].set_xticks(range(2))
    ax2[0].set_xticklabels(['1e-2','1e-1']) #[str(i) for i in phi_e[:3]])
     
    ax2[1].set_xticks(range(4))
    ax2[1].set_xticklabels(['1e0', '4e0', '8e0', '1e1']) #[str(i) for i in phi_e[3:7]])
    
    ax2[2].set_xticks(range(3))
    ax2[2].set_xticklabels(['1e2','1e3','1e4']) #[str(i) for i in phi_e[7:]])         
    
    ax2[0].set_yticklabels(['0']+[str(i) for i in np.arange(0,101,20)],fontsize=12)
elif dataset == 'MM':
    ax2[0].step(range(2),norm_tpe_nrg[:2],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_0 = ax2[0].twinx()
    twin2_0.step(range(2),tpe_param_tx[:2],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_0.set_ylim([-1,21])
    twin2_0.get_yaxis().set_ticklabels([])   
    
    ax2[1].step(range(5),norm_tpe_nrg[2:7],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_1 = ax2[1].twinx()
    twin2_1.step(range(5),tpe_param_tx[2:7],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_1.set_ylim([-1,21])
    twin2_1.get_yaxis().set_ticklabels([]) 

    ax2[2].step(range(3),norm_tpe_nrg[7:],where='post', \
            marker='x',linestyle='dashed',color='darkblue',markersize=10,linewidth=2)
    twin2_2 = ax2[2].twinx()
    twin2_2.step(range(3),tpe_param_tx[7:],where='post', \
            marker='x',linestyle='dashed',color='darkgreen',markersize=10,linewidth=2)  
    twin2_2.set_ylim([-1,21]) 

    ax2[0].set_xticks(range(2))
    ax2[0].set_xticklabels(['1e-3','1e-2'])
    
    ax2[1].set_xticks(range(5))
    ax2[1].set_xticklabels(['1e-1','5e-1','1e0','1e1','2e1'])
    
    ax2[2].set_xticks(range(3))
    ax2[2].set_xticklabels(['1e2','1e3','1e4'])
    
    ax2[0].set_yticklabels(['0']+[str(i) for i in np.arange(0,101,20)],fontsize=12)

twin2_2.set_ylabel('Saved Transmissions',fontsize=14)
twin2_2.tick_params(axis='y', colors='darkgreen')
twin2_2.yaxis.label.set_color('darkgreen')


for i in range(3):
    ax2[i].grid(True)

ax2[0].set_ylabel('Normalized Energy \n Consumption (%)',fontsize=14)

fig2.text(0.5, 0, r'Communication Energy Scaling $\phi^{E}$', ha='center',fontsize=14)
fig2.subplots_adjust(wspace=0.2)

fig2.savefig(cwd+'/nrg_plots/2nrg_norm_nrg'+dataset+'.png',dpi=1000,bbox_inches='tight')
fig2.savefig(cwd+'/nrg_plots/2nrg_norm_nrg'+dataset+'.pdf',dpi=1000,bbox_inches='tight')

    