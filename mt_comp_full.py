# -*- coding: utf-8 -*-
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
from optim_prob.div_utils.neural_nets import MLP, CNN, test_img_strain, GCNN,\
    feature_extract, class_classifier, GRL, test_img_gr
from utils.mt_utils import rescale_alphas, test_img_ttest, alpha_avg
from optim_prob.mnist_m import MNISTM

cwd = os.getcwd()
oargs = optim_parser()

np.random.seed(oargs.seed)
random.seed(oargs.seed)

if oargs.label_split == 0: 
    oargs.labels_type = 'iid'

# %% load in optimization and divergence results
psi_vals,alpha_vals = None, None
tval_dict = {'psi_val':psi_vals,'alpha_val':alpha_vals}

if oargs.nrg_mt == 0: 
    if oargs.dset_split == 0:
        for ie,entry in enumerate(tval_dict.keys()):
            with open(cwd+'/optim_prob/optim_results/'+entry+'/devices'+str(oargs.t_devices)+\
                      '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                        +'_'+oargs.dset_type\
                        +'_'+oargs.labels_type,'rb') as f:
                tval_dict[entry] = pk.load(f)
        ## load in the model parameters of all devices with labeled data
        with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn,'rb') as f:
            lmp = pk.load(f) #labeled model parameters        
    else:
        if oargs.dset_split == 1:
            pre = ''
        elif oargs.dset_split == 2:
            pre = 'total_'        
        for ie,entry in enumerate(tval_dict.keys()):
            with open(cwd+'/optim_prob/optim_results/'+entry+'/'+pre+'devices'+str(oargs.t_devices)+\
                      '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                        +'_'+oargs.split_type\
                        +'_'+oargs.labels_type,'rb') as f:
                tval_dict[entry] = pk.load(f)
        
        with open(cwd+'/optim_prob/source_errors/'+pre+'devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn,'rb') as f:
            lmp = pk.load(f) #labeled model parameters        
else: #load in the phi_e results
    if oargs.grad_rev == True:
        end2 = 'gr'
    else:
        end2 = ''
    if oargs.dset_split == 0:
        if oargs.dset_type == 'MM':
            end = '_base_6'
        else:
            end = ''
        for ie,entry in enumerate(tval_dict.keys()):
            with open(cwd+'/optim_prob/optim_results/'+entry+'/NRG_'+str(oargs.phi_e)+'_'+\
                      'devices'+str(oargs.t_devices)+\
                      '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                        +'_'+oargs.dset_type\
                        +'_'+oargs.labels_type+end+end2,'rb') as f:
                tval_dict[entry] = pk.load(f)
        ## load in the model parameters of all devices with labeled data
        with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn+end+end2,'rb') as f:
            lmp = pk.load(f) #labeled model parameters                 
    else:
        if oargs.dset_split == 1:
            pre = ''
        elif oargs.dset_split == 2:
            pre = 'total_'           
        if 'MM' in oargs.split_type:
            end = '_base_6'
        else:
            end = ''
        for ie,entry in enumerate(tval_dict.keys()):
            with open(cwd+'/optim_prob/optim_results/'+entry+'/NRG_'+str(oargs.phi_e)+'_'\
                      +pre+'devices'+str(oargs.t_devices)+\
                      '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                        +'_'+oargs.split_type\
                        +'_'+oargs.labels_type+end+end2,'rb') as f:
                tval_dict[entry] = pk.load(f)
        with open(cwd+'/optim_prob/source_errors/'+pre+'devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type\
                    +'_'+oargs.labels_type+'_modelparams_'+oargs.div_nn+end+end2,'rb') as f:
            lmp = pk.load(f) #labeled model parameters      

psi_vals = tval_dict['psi_val']
alpha_vals = tval_dict['alpha_val']

psi_vals = [int(np.round(j,0)) for j in psi_vals[len(psi_vals.keys())-1]]
s_alpha,t_alpha,ovr_alpha,s_pv,t_pv= rescale_alphas(psi_vals,alpha_vals)

## load in the device data characteristics
with open(cwd+'/optim_prob/data_div/devices'+str(oargs.t_devices)+\
          '_seed'+str(oargs.seed)+'_data_qty','rb') as f:
    data_qty = pk.load(f)
if oargs.dset_split == 0:
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
else:
    with open(cwd+'/optim_prob/data_div/'+pre+'devices'+str(oargs.t_devices)\
              +'_seed'+str(oargs.seed)\
            +'_'+oargs.div_nn\
            +'_'+oargs.split_type+'_'+oargs.labels_type+'_lpd','rb') as f:
        lpd = pk.load(f)
    with open(cwd+'/optim_prob/data_div/'+pre+'devices'+str(oargs.t_devices)\
              +'_seed'+str(oargs.seed)\
            +'_'+oargs.div_nn\
            +'_'+oargs.split_type+'_'+oargs.labels_type+'_dindexsets','rb') as f:
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
else: #if oargs.dset_split == 1: 
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
    
    if oargs.dset_split == 1:
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
    elif oargs.dset_split == 2: 
        with open(cwd+'/optim_prob/data_div/d2dset_devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed),'rb') as f:
            d2dset = pk.load(f)
        d_train_dict = {}
        
        if oargs.split_type == 'M+MM':
            d0,d1= d_m,d_mm
        elif oargs.split_type == 'M+U':
            d0,d1 = d_m,d_u
        elif oargs.split_type == 'MM+U':
            d0,d1 = d_mm,d_u
        else:
            raise TypeError('datasets exceed sims')
        
        for ind,dc in enumerate(np.where(d2dset==0)[0]):
            d_train_dict[dc] = d0
        for ind,dc in enumerate(np.where(d2dset==1)[0]):
            d_train_dict[dc] = d1
        d_train_dict = dict(sorted(d_train_dict.items()))


if oargs.div_comp == 'gpu':
    device = torch.device('cuda:'+str(oargs.div_gpu_num))
else:
    device = torch.device('cpu')
if oargs.grad_rev == True:
    feature_net_base = feature_extract().to(device)
    features_2_class_base = class_classifier().to(device)
    GRL_base = GRL().to(device)
    try:
        with open(cwd+'/optim_prob/optim_utils/fnet_base_w','rb') as f:
            fnet_base_w = pk.load(f)
        with open(cwd+'/optim_prob/optim_utils/f2c_base_w','rb') as f:
            f2c_base_w = pk.load(f)
        with open(cwd+'/optim_prob/optim_utils/GRL_base_w','rb') as f:
            GRL_base_w = pk.load(f)                
        feature_net_base.load_state_dict(fnet_base_w)     
        features_2_class_base.load_state_dict(f2c_base_w)
        GRL_base.load_state_dict(GRL_base_w)
    except:
        fnet_base_w = feature_net_base.state_dict()     
        f2c_base_w = features_2_class_base.state_dict()
        GRL_base_w = GRL_base.state_dict()
        with open(cwd+'/optim_prob/optim_utils/fnet_base_w','wb') as f:
            pk.dump(fnet_base_w,f)
        with open(cwd+'/optim_prob/optim_utils/f2c_base_w','wb') as f:
            pk.dump(f2c_base_w,f)
        with open(cwd+'/optim_prob/optim_utils/GRL_base_w','wb') as f:
            pk.dump(GRL_base_w,f)       
else:
    if oargs.div_nn == 'MLP':
        if oargs.dset_split < 2: 
            d_in = np.prod(d_train[0][0].shape)
        elif oargs.dset_split == 2:
            d_in = np.prod(d0[0][0].shape)
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

# %% build models at devices with labelled data
## special case here, where the optimization returns all devices with labelled data
## as sources [special case]
our_psi_vals = deepcopy(psi_vals)
our_smodels = {}
our_saccs = {}

wap_dict = {}
our_tmodels = {}
our_taccs = {}

if oargs.grad_rev == True:
    lmp1, lmp2 = {}, {}
    for tk in lmp.keys():
        lmp1[tk] = lmp[tk][0]
        lmp2[tk] = lmp[tk][1]
        
    for i,j in enumerate(our_psi_vals):
        if j == 0: # our algorithm's source
            our_smodels[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            our_smodels[i][0].load_state_dict(lmp1[i])
            our_smodels[i][1].load_state_dict(lmp2[i])            
            if oargs.dset_split < 2:
                our_saccs[i],_ = test_img_gr(our_smodels[i][0],our_smodels[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                our_saccs[i],_ = test_img_gr(our_smodels[i][0],our_smodels[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
        else:
            wap1 = alpha_avg(lmp1,ovr_alpha[:,i])
            wap2 = alpha_avg(lmp2,ovr_alpha[:,i])
            our_tmodels[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            our_tmodels[i][0].load_state_dict(wap1)
            our_tmodels[i][1].load_state_dict(wap2)
            if oargs.dset_split < 2:
                our_taccs[i],_ = test_img_gr(our_tmodels[i][0],our_tmodels[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                our_taccs[i],_ = test_img_gr(our_tmodels[i][0],our_tmodels[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
else:
    for i,j in enumerate(our_psi_vals):
        if j == 0: # our algorithm's source
            our_smodels[i] = deepcopy(start_net)
            our_smodels[i].load_state_dict(lmp[i])
            if oargs.dset_split < 2:
                our_saccs[i],_ = test_img_ttest(our_smodels[i],oargs.div_bs,\
                        d_train,d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                our_saccs[i],_ = test_img_ttest(our_smodels[i],oargs.div_bs,\
                        d_train_dict[i],d_dsets[i],device=device)
        else:
            wap = alpha_avg(lmp,ovr_alpha[:,i])
            wap_dict[i] = wap
            # test the resulting models
            our_tmodels[i] = deepcopy(start_net)
            our_tmodels[i].load_state_dict(wap)
            if oargs.dset_split < 2:
                our_taccs[i],_ = test_img_ttest(our_tmodels[i],oargs.div_bs,\
                        d_train,d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                our_taccs[i],_ = test_img_ttest(our_tmodels[i],oargs.div_bs,\
                        d_train_dict[i],d_dsets[i],device=device)

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

# %% determine the heurstic sources
## alg 1: g_avg_acc, all devices with source accuracies > avg become sources
## others become targets
h1_models = {}
h1_accs = {}
h1_nrg = 0

avg_saccs = np.average(list(our_saccs.values()))
h1_psi = deepcopy(psi_vals)
h1_th = list(np.where(list(our_saccs.values()) < avg_saccs)[0]) # targets
h1_s = list(np.where(list(our_saccs.values()) >= avg_saccs)[0])
h1_lmp = {}

for i in h1_th: #make target instead of source
    h1_psi[i] = 1
if oargs.grad_rev == True:
    h1_lmp1, h1_lmp2 = {}, {}
    for i,j in enumerate(h1_s): # grab the parameters in a dict
        h1_lmp1[i] = deepcopy(lmp1[j])
        h1_lmp2[i] = deepcopy(lmp2[j])
    
    for i,j in enumerate(h1_psi):
        if j == 1:
            # build models
            h1_alphas = np.round(np.random.dirichlet(np.ones(len(h1_s))),5)
            h1wp1 = alpha_avg(h1_lmp1,h1_alphas)
            h1wp2 = alpha_avg(h1_lmp2,h1_alphas)           
            h1_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            h1_models[i][0].load_state_dict(h1wp1)
            h1_models[i][1].load_state_dict(h1wp2)
            if oargs.dset_split < 2:        
                h1_accs[i],_ = test_img_gr(h1_models[i][0],h1_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                h1_accs[i],_ = test_img_gr(h1_models[i][0],h1_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
            tmp_c2d_rates = d2d_tx_rates[:,i][h1_s]
            h1_nrg += mt_nrg_calc(h1_alphas,tmp_c2d_rates)
        elif j == 0 and oargs.grad_rev == True:
            h1_accs[i] = our_saccs[i]
else:
    for i,j in enumerate(h1_s): # grab the parameters in a dict
        h1_lmp[i] = deepcopy(lmp[j])
    
    for i,j in enumerate(h1_psi):
        if j == 1:
            # build models
            h1_alphas = np.round(np.random.dirichlet(np.ones(len(h1_s))),5)
            h1wp = alpha_avg(h1_lmp,h1_alphas)
            h1_models[i] = deepcopy(start_net)
            h1_models[i].load_state_dict(h1wp)
            if oargs.dset_split < 2:        
                h1_accs[i],_ = test_img_ttest(h1_models[i],oargs.div_bs,\
                            d_train,d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                h1_accs[i],_ = test_img_ttest(h1_models[i],oargs.div_bs,\
                            d_train_dict[i],d_dsets[i],device=device)
            tmp_c2d_rates = d2d_tx_rates[:,i][h1_s]
            h1_nrg += mt_nrg_calc(h1_alphas,tmp_c2d_rates)

## alg 2: greatest accuracy
h2_models = {}
h2_accs = {}
h2_nrg = 0

h2_psi = deepcopy(psi_vals)
h2_s = np.argmax(list(our_saccs.values()))
h2_s = [h2_s]
h2_th = list(np.where(np.round(list(our_saccs.values()),2) < \
                np.round(our_saccs[h2_s[0]],2))[0])
h2_lmp = {}
for i in h2_th:
    h2_psi[i] = 1
    
if oargs.grad_rev == True:
    h2_lmp1, h2_lmp2 = {}, {}
    for i,j in enumerate(h2_s): # grab the parameters in a dict
        h2_lmp1[i] = deepcopy(lmp1[j])
        h2_lmp2[i] = deepcopy(lmp2[j])
    for i,j in enumerate(h2_psi):
        if j == 1:
            # build models
            h2_alphas = np.round(np.random.dirichlet(np.ones(len(h2_s))),5)
            h2wp1 = alpha_avg(h2_lmp1,h2_alphas)
            h2wp2 = alpha_avg(h2_lmp2,h2_alphas)            
            h2_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            h2_models[i][0].load_state_dict(h2wp1)
            h2_models[i][1].load_state_dict(h2wp2)
            if oargs.dset_split < 2:
                h2_accs[i],_ = test_img_gr(h2_models[i][0],h2_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                h2_accs[i],_ = test_img_gr(h2_models[i][0],h2_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
            tmp_c2d_rates = d2d_tx_rates[:,i][h2_s]
            h2_nrg += mt_nrg_calc(h2_alphas,tmp_c2d_rates)
        elif j == 0 and oargs.grad_rev == True:
            h2_accs[i] = our_saccs[i]
else:
    for i,j in enumerate(h2_s):
        h2_lmp[i] = deepcopy(lmp[j])
    for i,j in enumerate(h2_psi):
        if j == 1:
            h2_alphas = np.round(np.random.dirichlet(np.ones(len(h2_s))),5)
            h2wp = alpha_avg(h2_lmp,h2_alphas)
            h2_models[i] = deepcopy(start_net)
            h2_models[i].load_state_dict(h2wp)
            if oargs.dset_split < 2:        
                h2_accs[i],_ = test_img_ttest(h2_models[i],oargs.div_bs,\
                            d_train,d_dsets[i],device=device)      
            elif oargs.dset_split == 2:
                h2_accs[i],_ = test_img_ttest(h2_models[i],oargs.div_bs,\
                            d_train_dict[i],d_dsets[i],device=device)
            tmp_c2d_rates = d2d_tx_rates[:,i][h2_s]
            h2_nrg += mt_nrg_calc(h2_alphas,tmp_c2d_rates)       


## alg 3: random source determination from devices with labelled datasets
## special case, as our algorithm (for this case) uses all labelled devices
## as sources
r_models = {}
r_accs = {}
r_s_accs = {}
r_nrg = 0

r_psi = np.array(deepcopy(psi_vals))
r_s_init = list(np.where(r_psi == 0)[0])
tt = np.random.rand(len(r_s_init))
r_s_change = list(np.where(tt >= 0.75)[0]) #then a target and not a source
for i in r_s_change:
    r_s_init.pop(r_s_init.index(i))
    r_psi[i] = 1
r_s = r_s_init
r_lmp = {}

    
if oargs.grad_rev == True:
    r_lmp1, r_lmp2 = {}, {}
    for i,j in enumerate(r_s): # grab the parameters in a dict
        r_lmp1[i] = deepcopy(lmp1[j])
        r_lmp2[i] = deepcopy(lmp2[j])
    for i,j in enumerate(r_psi):
        if j == 1:
            # build models (random weights)
            r_alphas = np.round(np.random.dirichlet(np.ones(len(r_s))),5)
            rwp1 = alpha_avg(r_lmp1,r_alphas)
            rwp2 = alpha_avg(r_lmp2,r_alphas)
            r_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            r_models[i][0].load_state_dict(rwp1)
            r_models[i][1].load_state_dict(rwp2)
            if oargs.dset_split < 2:
                r_accs[i],_ = test_img_gr(r_models[i][0],r_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                r_accs[i],_ = test_img_gr(r_models[i][0],r_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
            tmp_c2d_rates = d2d_tx_rates[:,i][r_s]
            r_nrg += mt_nrg_calc(r_alphas,tmp_c2d_rates)
        elif j == 0 and oargs.grad_rev == True:
            r_accs[i] = our_saccs[i]
else:
    for i,j in enumerate(r_s):
        r_lmp[i] = deepcopy(lmp[j])
    
    for i,j in enumerate(r_psi):
        if j == 1:
            # build models (random weights)
            r_alphas = np.round(np.random.dirichlet(np.ones(len(r_s))),5)
            r_wp = alpha_avg(r_lmp,r_alphas)
            r_models[i] = deepcopy(start_net)
            r_models[i].load_state_dict(r_wp)
            if oargs.dset_split < 2:          
                r_accs[i],_ = test_img_ttest(r_models[i],oargs.div_bs,\
                            d_train,d_dsets[i],device=device)   
            elif oargs.dset_split == 2:
                r_accs[i],_ = test_img_ttest(r_models[i],oargs.div_bs,\
                            d_train_dict[i],d_dsets[i],device=device)
            tmp_c2d_rates = d2d_tx_rates[:,i][r_s]
            r_nrg += mt_nrg_calc(r_alphas,tmp_c2d_rates)
            

# %% save the results
import pandas as pd
acc_df = pd.DataFrame()
accs_vec = [list(r_accs.values()),list(h1_accs.values()),list(h2_accs.values())]
accs_vec_lens = [len(i) for i in accs_vec]
df_max_len = np.max(accs_vec_lens)
df_max_len_ind = np.argmax(accs_vec_lens)
for i,j in enumerate(accs_vec):
    if len(j) < df_max_len:
        tarr = np.empty(df_max_len - len(j))
        tarr[:] = np.nan
        j += tarr.tolist()
        accs_vec[i] = j

acc_df['rng'] = accs_vec[0]
acc_df['geq_avg_acc'] = accs_vec[1]
acc_df['max_acc'] = accs_vec[2]

nrg_df = pd.DataFrame()
nrg_df['rng'] = [r_nrg]
nrg_df['geq_avg_acc'] = [h1_nrg]
nrg_df['max_acc'] = [h2_nrg]
if oargs.nrg_mt == 0:
    if oargs.dset_split == 0: # only one dataset
        acc_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/seed_'+str(oargs.seed)\
                +'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+'_acc.csv')
        nrg_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/seed_'+str(oargs.seed)\
                +'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+'_nrg.csv')
    else:
        acc_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/seed_'+str(oargs.seed)\
                +'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+'_acc.csv')
        nrg_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/seed_'+str(oargs.seed)\
                +'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+'_nrg.csv')

else: ## adjust file name with nrg
    if oargs.dset_split == 0: # only one dataset
        acc_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/NRG'+str(oargs.phi_e)\
                  +'_seed_'+str(oargs.seed)+'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+end+'_acc.csv')
        nrg_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/NRG'+str(oargs.phi_e)\
                  +'_seed_'+str(oargs.seed)+'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+end+end2+'_nrg.csv')
    else:
        acc_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +pre+'seed_'+str(oargs.seed)+'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+end+end2+'_acc.csv')
        nrg_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/NRG'+str(oargs.phi_e)+'_'\
                  +pre+'seed_'+str(oargs.seed)+'_st_det_'+oargs.labels_type \
                  +'_'+oargs.div_nn+end+end2+'_nrg.csv') 



