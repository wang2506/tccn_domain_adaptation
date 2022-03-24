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
from optim_prob.div_utils.neural_nets import MLP, CNN, test_img_strain
from utils.mt_utils import rescale_alphas, test_img_ttest, alpha_avg

cwd = os.getcwd()
oargs = optim_parser()

np.random.seed(oargs.seed)
random.seed(oargs.seed)

if oargs.label_split == 0: #iid
    oargs.labels_type = 'iid'

# %% load in optimization and divergence results
if oargs.dset_split == 0:
    with open(cwd+'/optim_prob/optim_results/psi_val/devices'+str(oargs.t_devices)+\
              '_seed'+str(oargs.seed)+'_'+oargs.dset_type\
                +'_'+oargs.labels_type,'rb') as f:
        psi_vals = pk.load(f)
    
    with open(cwd+'/optim_prob/optim_results/alpha_val/devices'+str(oargs.t_devices)+\
              '_seed'+str(oargs.seed)+'_'+oargs.dset_type\
                +'_'+oargs.labels_type,'rb') as f:
        alpha_vals = pk.load(f)
        
    ## load in the model parameters of all devices with labeled data
    with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
              '_seed'+str(oargs.seed)+'_'+oargs.dset_type\
                +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn,'rb') as f:
        lmp = pk.load(f) #labeled model parameters        
else:
    with open(cwd+'/optim_prob/optim_results/psi_val/devices'+str(oargs.t_devices)+\
              '_seed'+str(oargs.seed)+'_'+oargs.split_type\
                +'_'+oargs.labels_type,'rb') as f:
        psi_vals = pk.load(f)
    
    with open(cwd+'/optim_prob/optim_results/alpha_val/devices'+str(oargs.t_devices)+\
              '_seed'+str(oargs.seed)+'_'+oargs.split_type\
                +'_'+oargs.labels_type,'rb') as f:
        alpha_vals = pk.load(f)    
    
    with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
              '_seed'+str(oargs.seed)+'_'+oargs.split_type\
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
    +'_'+oargs.dset_type+'_'+oargs.labels_type+'_lpd','rb') as f:
    lpd = pk.load(f)
with open(cwd+'/optim_prob/data_div/devices'+str(oargs.t_devices)\
          +'_seed'+str(oargs.seed)\
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
    else:
        raise TypeError('Dataset exceeds sims')
elif oargs.dset_split == 1: # TODO
    if oargs.dset_type == 'M+S':
        print('Using MNIST + SVHN')
    elif oargs.dset_type == 'M+U':
        print('Using MNIST + USPS')
    elif oargs.dset_type == 'S+U':
        print('Using SVHN + USPS')
    elif oargs.dset_type == 'A':
        print('Using MNIST + SVHN + USPS')
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

# %% build model + transfer to targets + record results
wap_dict = {}
target_models = {}
rt_models = {}
h1_models = {} #heuristic ratio/scale by data qty 
h2_models = {} #heuristic - uniform

target_accs = {}
rt_accs = {}
h1_accs = {}
h2_accs = {}

source_models = {}
source_accs = {}

for i,j in enumerate(psi_vals):
    if j == 1:
        wap = alpha_avg(lmp,ovr_alpha[:,i]) #w_avg_params
        wap_dict[i] = wap
        
        # test the resulting models
        target_models[i] = deepcopy(start_net)
        target_models[i].load_state_dict(wap)
        target_accs[i],_ = test_img_ttest(target_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        # build random models + test them
        rwp = alpha_avg(lmp,np.round(np.random.dirichlet(np.ones(len(s_pv))),5))
        
        rt_models[i] = deepcopy(start_net)
        rt_models[i].load_state_dict(rwp)
        rt_accs[i],_ = test_img_ttest(rt_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        # heuristic qty
        h1wp = alpha_avg(lmp,np.round(np.array(data_qty)[s_pv]/max(data_qty),5))
        
        h1_models[i] = deepcopy(start_net)
        h1_models[i].load_state_dict(h1wp)
        h1_accs[i],_ = test_img_ttest(h1_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)
        
        # heuristic max source labels distribution
        h2wp = alpha_avg(lmp,1/len(s_pv) * np.ones(len(s_pv)))
        
        h2_models[i] = deepcopy(start_net)
        h2_models[i].load_state_dict(h2wp)
        h2_accs[i],_ = test_img_ttest(h2_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)        
        
    else:
        source_models[i] = deepcopy(start_net)
        source_models[i].load_state_dict(lmp[i])
        source_accs[i],_ = test_img_ttest(source_models[i],oargs.div_bs,d_train,d_dsets[i],device=device)

# %% save the results
if oargs.dset_split == 0: # only one dataset
    with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
              +'_full_target','wb') as f:
        pk.dump(target_accs,f)

    with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
              +'_full_rng','wb') as f:
        pk.dump(rt_accs,f)
    
    with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
              +'_full_h1','wb') as f:
        pk.dump(h1_accs,f)

    with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
              +'_full_h2','wb') as f:
        pk.dump(h2_accs,f)        
    
    with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
              +'_full_source','wb') as f:
        pk.dump(source_accs,f)  
else:
    with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
              +'_full_target','wb') as f:
        pk.dump(target_accs,f)

    with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
              +'_full_rng','wb') as f:
        pk.dump(rt_accs,f)
        
    with open(cwd+'/mt_results/'+oargs.split_type+'/'+oargs.labels_type \
              +'_full_h1','wb') as f:
        pk.dump(h1_accs,f)

    with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
              +'_full_h2','wb') as f:
        pk.dump(h2_accs,f)        
    
    with open(cwd+'/mt_results/'+oargs.dset_type+'/'+oargs.labels_type \
              +'_full_source','wb') as f:
        pk.dump(source_accs,f)      









