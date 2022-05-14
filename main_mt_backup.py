# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:52:38 2022

@author: ch5b2
"""

import os
import numpy as np
from copy import deepcopy
import pickle as pk
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

from optim_prob.optim_utils.optim_parser import optim_parser
from optim_prob.div_utils.neural_nets import MLP, CNN, test_img_strain, GCNN
from utils.mt_utils import rescale_alphas, test_img_ttest, alpha_avg
from optim_prob.mnist_m import MNISTM

cwd = os.getcwd()
oargs = optim_parser()

np.random.seed(oargs.seed)
random.seed(oargs.seed)

if oargs.label_split == 0: #iid
    oargs.labels_type = 'iid'

# %% Some notes
# This file contains the model transfer analysis and evaluation for 
# our optimization results. 
# We compare ours and 3 algos (random, heuristic 1 and 2) that 
# find weights (\alpha).
# These three alternative algos use the source/target determination from
# our optimization solver.
#
# For a full comparison of weights + source/target determination, see
# mt_comp_full.
#
# %% load in optimization and divergence results
if oargs.nrg_mt == 0: 
    if oargs.dset_split == 0:
        with open(cwd+'/optim_prob/optim_results/psi_val/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type,'rb') as f:
            psi_vals = pk.load(f)
        
        with open(cwd+'/optim_prob/optim_results/alpha_val/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type,'rb') as f:
            alpha_vals = pk.load(f)
            
        ## load in the model parameters of all devices with labeled data
        with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn,'rb') as f:
            lmp = pk.load(f) #labeled model parameters        
    else:
        with open(cwd+'/optim_prob/optim_results/psi_val/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type,'rb') as f:
            psi_vals = pk.load(f)
        
        with open(cwd+'/optim_prob/optim_results/alpha_val/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type,'rb') as f:
            alpha_vals = pk.load(f)    
        
        with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn,'rb') as f:
            lmp = pk.load(f) #labeled model parameters        
else: #load in the modified phi_e results
    if oargs.dset_split == 0:
        with open(cwd+'/optim_prob/optim_results/psi_val/NRG_'+str(oargs.phi_e)+'_'+\
                  'devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type,'rb') as f:
            psi_vals = pk.load(f)
        
        with open(cwd+'/optim_prob/optim_results/alpha_val/NRG_'+str(oargs.phi_e)+'_'+\
                  'devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type,'rb') as f:
            alpha_vals = pk.load(f)
            
        ## load in the model parameters of all devices with labeled data
        with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn,'rb') as f:
            lmp = pk.load(f) #labeled model parameters        
    else:
        with open(cwd+'/optim_prob/optim_results/psi_val/NRG_'+str(oargs.phi_e)+'_'+\
                  'devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type,'rb') as f:
            psi_vals = pk.load(f)
        
        with open(cwd+'/optim_prob/optim_results/alpha_val/NRG_'+str(oargs.phi_e)+'_'+\
                  'devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type,'rb') as f:
            alpha_vals = pk.load(f)    
        
        with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn,'rb') as f:
            lmp = pk.load(f) #labeled model parameters      

psi_vals = [int(np.round(j,0)) for j in psi_vals[len(psi_vals.keys())-1]]
s_alpha,t_alpha,ovr_alpha,s_pv,t_pv= rescale_alphas(psi_vals,alpha_vals)

## load in the device data characteristics
with open(cwd+'/optim_prob/data_div/devices'+str(oargs.t_devices)+\
          '_seed'+str(oargs.seed)+'_data_qty','rb') as f:
    data_qty = pk.load(f)

with open(cwd+'/optim_prob/data_div/devices'+str(oargs.t_devices)\
          +'_seed'+str(oargs.seed)\
        +'_'+oargs.div_nn\
        +'_'+oargs.dset_type+'_'+oargs.labels_type+'_lpd','rb') as f:
    lpd = pk.load(f)
with open(cwd+'/optim_prob/data_div/devices'+str(oargs.t_devices)\
          +'_seed'+str(oargs.seed)\
        +'_'+oargs.div_nn\
        +'_'+oargs.dset_type+'_'+oargs.labels_type+'_dindexsets','rb') as f:
    d_dsets = pk.load(f)    

# %% load in datasets
if oargs.dset_split == 0:
    if oargs.dset_type == 'M':
        print('Using MNIST \n')
        d_train = torchvision.datasets.MNIST(cwd+'/data/',train=True,download=True,\
                        transform=transforms.ToTensor())
    elif oargs.dset_type == 'S': #needs scipy
        print('Using SVHN \n')
        tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),\
                        transforms.Grayscale(),transforms.CenterCrop(28)])
        d_train = torchvision.datasets.SVHN(cwd+'/data/svhn',split='train',download=True,\
                        transform=tx_dat)
        d_train.targets = d_train.labels
        #http://ufldl.stanford.edu/housenumbers/
    elif oargs.dset_type == 'U':
        print('Using USPS \n')
        tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])
        try: 
            d_train = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                            transform=tx_dat)
        except:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context            
            d_train = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                            transform=tx_dat)    
    elif oargs.dset_type == 'MM':
        print('Using MNIST-M \n')
        tx_dat =  torchvision.transforms.Compose([transforms.ToTensor()])
        d_train = MNISTM(cwd+'/data/',train=True,download=True,\
                         transform=tx_dat)                    
    else:
        raise TypeError('Dataset exceeds sims')
elif oargs.dset_split == 1: 
    tx_m = torchvision.transforms.Compose([transforms.ToTensor()])
    tx_mm = torchvision.transforms.Compose([transforms.ToTensor(),\
                transforms.Grayscale()])
    tx_u = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])    
    
    d_m = torchvision.datasets.MNIST(cwd+'/data/',train=True,download=True,\
                    transform=tx_m)
    d_mm = MNISTM(cwd+'/data/',train=True,download=True,\
                     transform=tx_mm)        
    try: 
        d_u = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                        transform=tx_u)
    except:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context            
        d_u = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                        transform=tx_u)
    d_u.targets = torch.tensor(d_u.targets)    
    
    if oargs.split_type == 'M+MM':
        print('Using MNIST + MNIST-M')
        d_train = d_m+d_mm
        d_train.targets = torch.concat([d_m.targets,d_mm.targets])
    elif oargs.split_type == 'M+U':
        print('Using MNIST + USPS')
        d_train = d_m+d_u
        d_train.targets = torch.concat([d_m.targets,d_u.targets])
    elif oargs.split_type == 'MM+U':
        print('Using MNIST-M + USPS')
        d_train = d_mm+d_u       
        d_train.targets = torch.concat([d_mm.targets,d_u.targets])
    elif oargs.split_type == 'A':
        print('Using MNIST + MNIST-M + USPS')
        d_train = d_m+d_mm+d_u
        d_train.targets = torch.concat([d_m.targets,d_mm.targets,d_u.targets])
    else:
        raise TypeError('Datasets exceed sims')


if oargs.div_comp == 'gpu':
    device = torch.device('cuda:'+str(oargs.div_gpu_num))
else:
    device = torch.device('cpu')
if oargs.div_nn == 'MLP':
    d_in = np.prod(d_train[0][0].shape)
    d_h = 64
    d_out = 10
    start_net = MLP(d_in,d_h,d_out).to(device)

    try:
        with open(cwd+'/optim_prob/optim_utils/MLP_start_w','rb') as f:
            start_w = pk.load(f)
        start_net.load_state_dict(start_w)
    except:
        start_w = start_net.state_dict()
        with open(cwd+'/optim_prob/optim_utils/MLP_start_w','wb') as f:
            pk.dump(start_w,f)
elif oargs.div_nn == 'CNN':
    nchannels = 1
    nclasses = 10
    start_net = CNN(nchannels,nclasses).to(device)
    try:
        with open(cwd+'/optim_prob/optim_utils/CNN_start_w','rb') as f:
            start_w = pk.load(f)
        start_net.load_state_dict(start_w)
    except:
        start_w = start_net.state_dict()
        with open(cwd+'/optim_prob/optim_utils/CNN_start_w','wb') as f:
            pk.dump(start_w,f)
else:
    nchannels = 1
    nclasses = 10
    start_net = GCNN(nchannels,nclasses).to(device)    

# %% energy compute fxn + load in vars
with open(cwd+'/optim_prob/nrg_constants/devices'+str(oargs.t_devices)\
    +'_d2dtxrates','rb') as f:
    d2d_tx_rates = pk.load(f)
with open(cwd+'/optim_prob/nrg_constants/devices'+str(oargs.t_devices)\
    +'_txpowers','rb') as f:
    tx_powers = pk.load(f)   

def mt_nrg_calc(tc_alpha,c2d_rates,tx_pow=tx_powers,M=oargs.p2bits):
    param_2_bits = M
    
    # calculate energy used for model transferring
    ctx_nrg = 0
    for ind_ca,ca in enumerate(tc_alpha):
        if ca > 1e-3:
            ctx_nrg += param_2_bits/c2d_rates[ind_ca] * tx_powers[ind_ca] #* ca
    
    return ctx_nrg #current tx energy

# %% 
def calc_odeg(ovr_alpha=ovr_alpha,psi_vals=psi_vals):
    sources = 0
    num_tx = 0
    for i,j in enumerate(psi_vals):
        if j == 0:
            sources += 1
            num_tx += len(np.where(ovr_alpha[i,:] > 1e-3)[0])    
    return round(num_tx/sources)

def calc_sm_alphas(deg,ovr_alpha=ovr_alpha,psi_vals=psi_vals,oargs=oargs):
    tsm_alphas = deepcopy(ovr_alpha)
    for i,j in enumerate(psi_vals):
        if j == 0:
            temp_alpha_vec = np.zeros_like(tsm_alphas[i,:])
            for td in range(deg):
                t_ind = np.argmax(tsm_alphas[i,:])
                while t_ind < oargs.u_devices:
                    tsm_alphas[i,:][t_ind] = -1
                    t_ind = np.argmax(tsm_alphas[i,:])
                temp_alpha_vec[t_ind] = np.random.rand()
                #1/deg #everyone is equally weighted #np.max(tsm_alphas[i,:])+1e-6
                tsm_alphas[i,:][t_ind] = -1
    
            tsm_alphas[i,:] = temp_alpha_vec
            
    # normalize over columns
    for i,j in enumerate(psi_vals):
        if j == 1:
            tsm_alphas[:,j] /= sum(tsm_alphas[:,j])
            tsm_alphas[:,j] = np.round(tsm_alphas[i,:],2)
    return tsm_alphas
    
# %% build model + transfer to targets + record results
wap_dict = {}
target_models = {}
rt_models = {}
h1_models = {} #heuristic ratio/scale by data qty 
h2_models = {} #heuristic - uniform
oo_models = {} #single source to single target
sm_models = {} #single source to multi-target

target_accs = {}
rt_accs = {}
h1_accs = {}
h2_accs = {}
oo_accs = {} #single source to single target
sm_accs = {} #single source to multi-target

source_models = {}
source_accs = {}

our_nrg = 0
r_nrg = 0
h1_nrg = 0
h2_nrg = 0
oo_nrg = 0
sm_nrg = 0

occupied_sources = np.zeros_like(np.where(np.array(psi_vals)==0)[0])
oo_alpha = deepcopy(ovr_alpha)
# sm_alpha = deepcopy(ovr_alpha) #each source sends its three highest ratios
avg_odeg = calc_odeg()
sm_alpha = calc_sm_alphas(avg_odeg)

# input('a')
for i,j in enumerate(psi_vals):
    if j == 1:
        wap = alpha_avg(lmp,ovr_alpha[:,i]) #w_avg_params
        wap_dict[i] = wap
        
        # test the resulting models
        target_models[i] = deepcopy(start_net)
        target_models[i].load_state_dict(wap)
        target_accs[i],_ = test_img_ttest(target_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        # build random models + test them
        r_alpha = np.round(np.random.dirichlet(np.ones(len(s_pv))),5)
        rwp = alpha_avg(lmp,r_alpha)
        
        rt_models[i] = deepcopy(start_net)
        rt_models[i].load_state_dict(rwp)
        rt_accs[i],_ = test_img_ttest(rt_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        # heuristic qty
        h1_alpha = np.round(np.array(data_qty)[s_pv]/max(data_qty),5)
        h1wp = alpha_avg(lmp,h1_alpha)
        
        h1_models[i] = deepcopy(start_net)
        h1_models[i].load_state_dict(h1wp)
        h1_accs[i],_ = test_img_ttest(h1_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        # heuristic - uniform ratios
        h2_alpha = 1/len(s_pv) * np.ones(len(s_pv))
        h2wp = alpha_avg(lmp,h2_alpha)
        
        h2_models[i] = deepcopy(start_net)
        h2_models[i].load_state_dict(h2wp)
        h2_accs[i],_ = test_img_ttest(h2_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)        
        
        ## one-to-one and one-to-many
        tta = oo_alpha[:,i][:oargs.l_devices]
        t_ind = np.argmax(tta)
        while occupied_sources[t_ind] == 1: #take next best one
            tta[t_ind] = -1
            t_ind = np.argmax(tta)
        occupied_sources[t_ind] = 1
        # print('oo')
        # print(occupied_sources)
        # print(t_ind)
        
        oo_alpha2 = np.zeros_like(ovr_alpha[:,i])
        oo_alpha2[t_ind] = 1
        oo_wp = alpha_avg(lmp,oo_alpha2)
        
        oo_models[i] = deepcopy(start_net)
        oo_models[i].load_state_dict(oo_wp)
        oo_accs[i],_ = test_img_ttest(oo_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        # one-to-many [approximate avg out degree]
        sm_wp = alpha_avg(lmp,sm_alpha[:,i])
        
        sm_models[i] = deepcopy(start_net)
        sm_models[i].load_state_dict(sm_wp)
        sm_accs[i],_ = test_img_ttest(sm_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        ## compute energies
        tmp_c2d_rates = d2d_tx_rates[:,i]
        our_nrg += mt_nrg_calc(ovr_alpha[:,i],tmp_c2d_rates)
        r_nrg += mt_nrg_calc(r_alpha,tmp_c2d_rates)
        h1_nrg += mt_nrg_calc(h1_alpha,tmp_c2d_rates)
        h2_nrg += mt_nrg_calc(h2_alpha,tmp_c2d_rates)
        oo_nrg += mt_nrg_calc(oo_alpha2,tmp_c2d_rates)
        sm_nrg += mt_nrg_calc(sm_alpha[:,i],tmp_c2d_rates)
    else:
        source_models[i] = deepcopy(start_net)
        source_models[i].load_state_dict(lmp[i])
        source_accs[i],_ = test_img_ttest(source_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)

        

# %% save the results
import pandas as pd
acc_df = pd.DataFrame()
acc_df['ours'] = list(target_accs.values())
acc_df['rng'] = list(rt_accs.values())
acc_df['max_qty'] = list(h1_accs.values())
acc_df['unif_ratio'] = list(h2_accs.values())
acc_df['o2o'] = list(oo_accs.values())
acc_df['o2m'] = list(sm_accs.values())
acc_df['source'] = list(source_accs.values())

nrg_df = pd.DataFrame()
nrg_df['ours'] = list(our_nrg.values())
nrg_df['rng'] = list(r_nrg.values())
nrg_df['max_qty'] = list(h1_nrg.values())
nrg_df['unif_ratio'] = list(h2_nrg.values())
nrg_df['o2o'] = list(oo_nrg.values())
nrg_df['o2m'] = list(sm_nrg.values())

if oargs.nrg_mt == 0:
    if oargs.dset_split == 0: # only one dataset
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_target','wb') as f:
            pk.dump(target_accs,f)
    
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_rng','wb') as f:
            pk.dump(rt_accs,f)
        
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_h1','wb') as f:
            pk.dump(h1_accs,f)
    
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_h2','wb') as f:
            pk.dump(h2_accs,f)        
        
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_source','wb') as f:
            pk.dump(source_accs,f)  
        
        ## save energies
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_target_nrg','wb') as f:
            pk.dump(our_nrg,f)
    
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_rng_nrg','wb') as f:
            pk.dump(r_nrg,f)
        
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_h1_nrg','wb') as f:
            pk.dump(h1_nrg,f)
    
        with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_h2_nrg','wb') as f:
            pk.dump(h2_nrg,f)               
    else:
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_target','wb') as f:
            pk.dump(target_accs,f)
    
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_rng','wb') as f:
            pk.dump(rt_accs,f)
            
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_h1','wb') as f:
            pk.dump(h1_accs,f)
    
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_h2','wb') as f:
            pk.dump(h2_accs,f)        
        
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_source','wb') as f:
            pk.dump(source_accs,f)      
    
        ## save energies
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_target_nrg','wb') as f:
            pk.dump(our_nrg,f)
    
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_rng_nrg','wb') as f:
            pk.dump(r_nrg,f)
        
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_h1_nrg','wb') as f:
            pk.dump(h1_nrg,f)
    
        with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_h2_nrg','wb') as f:
            pk.dump(h2_nrg,f)   
else: ## adjust file name with nrg
    if oargs.dset_split == 0: # only one dataset
        with open(cwd+'/mt_results/'+oargs.dset_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_target','wb') as f:
            pk.dump(target_accs,f)      
        
        with open(cwd+'/mt_results/'+oargs.dset_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_source','wb') as f:
            pk.dump(source_accs,f)  
            
        ## save energies
        with open(cwd+'/mt_results/'+oargs.dset_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_target_nrg','wb') as f:
            pk.dump(our_nrg,f)
              
    else:
        with open(cwd+'/mt_results/'+oargs.split_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_target','wb') as f:
            pk.dump(target_accs,f)

        with open(cwd+'/mt_results/'+oargs.split_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_full_source','wb') as f:
            pk.dump(source_accs,f)      
    
        ## save energies
        with open(cwd+'/mt_results/'+oargs.split_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +oargs.labels_type \
                  +'_'+oargs.div_nn\
                +'_target_nrg','wb') as f:
            pk.dump(our_nrg,f)
    
    






