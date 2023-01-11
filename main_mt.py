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
                    +'_'+oargs.dset_type+'_'+oargs.labels_type\
                    +'_modelparams_'+oargs.div_nn,'rb') as f:
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
    
    if oargs.fl == True:
        prefl = 'fl'
    else:
        prefl = ''
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
                        +'_'+oargs.labels_type+prefl+end+end2,'rb') as f:
                tval_dict[entry] = pk.load(f)
        ## load in the model parameters of all devices with labeled data
        with open(cwd+'/optim_prob/source_errors/devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.dset_type+'_'+oargs.labels_type\
                    +'_'+prefl+'_modelparams_'+oargs.div_nn+end+end2,'rb') as f:
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
                        +'_'+oargs.labels_type+prefl+end+end2,'rb') as f:
                tval_dict[entry] = pk.load(f)
        with open(cwd+'/optim_prob/source_errors/'+pre+'devices'+str(oargs.t_devices)+\
                  '_seed'+str(oargs.seed)+'_'+oargs.div_nn\
                    +'_'+oargs.split_type+'_'+oargs.labels_type\
                    +'_'+prefl+'_modelparams_'+oargs.div_nn+end+end2,'rb') as f:
            lmp = pk.load(f) #labeled model parameters      

psi_vals = tval_dict['psi_val']
alpha_vals = tval_dict['alpha_val']

psi_vals = [int(np.round(j,0)) for j in psi_vals[len(psi_vals.keys())-1]]
s_alpha,t_alpha,ovr_alpha,s_pv,t_pv= rescale_alphas(psi_vals,alpha_vals)

## load in the device data characteristics
with open(cwd+'/optim_prob/data_div/devices'+str(oargs.t_devices)+\
          '_seed'+str(oargs.seed)+'_data_qty'+'_'+oargs.avg_size,'rb') as f:
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
        os_append = 'MLP_start_w'
    elif oargs.div_nn == 'CNN':
        nchannels = 1
        nclasses = 10
        start_net = CNN(nchannels,nclasses).to(device)
        os_append = 'CNN_start_w'
    elif oargs.div_nn == 'GCNN':
        nchannels = 1
        nclasses = 10
        start_net = GCNN(nchannels,nclasses).to(device)    
        os_append = 'GCNN_start_w'
    try:
        with open(cwd+'/optim_prob/optim_utils/{}'.format(os_append),'rb') as f:
            start_w = pk.load(f)
        start_net.load_state_dict(start_w)
    except:
        start_w = start_net.state_dict()
        with open(cwd+'/optim_prob/optim_utils/{}'.format(os_append),'wb') as f:
            pk.dump(start_w,f)

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
    return int(np.ceil(num_tx/sources))

def calc_sm_alphas(deg,ovr_alpha=ovr_alpha,psi_vals=psi_vals,oargs=oargs):
    tsm_alphas = deepcopy(ovr_alpha)
    for i,j in enumerate(psi_vals):
        if j == 0:
            temp_alpha_vec = np.zeros_like(tsm_alphas[i,:])
            td_vec = oargs.l_devices+np.array(random.sample(range(oargs.u_devices),deg))
            for td in td_vec:
                temp_alpha_vec[td] = np.random.rand()
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
fl_models = {}

target_accs = {}
rt_accs = {}
h1_accs = {}
h2_accs = {}
oo_accs = {} #single source to single target
sm_accs = {} #single source to multi-target
fl_accs = {} #comparison to FL

source_models = {}
source_accs = {}

our_nrg = 0
r_nrg = 0
h1_nrg = 0
h2_nrg = 0

if oargs.grad_rev == True:
    lmp1, lmp2 = {}, {}
    for tk in lmp.keys():
        lmp1[tk] = lmp[tk][0]
        lmp2[tk] = lmp[tk][1]
for i,j in enumerate(psi_vals):
    if j == 1:
        if oargs.grad_rev == True:
            # ours
            wap1 = alpha_avg(lmp1,ovr_alpha[:,i]) 
            wap2 = alpha_avg(lmp2,ovr_alpha[:,i]) 
            target_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            target_models[i][0].load_state_dict(wap1)
            target_models[i][1].load_state_dict(wap2)
            
            # random
            r_alpha = np.round(np.random.dirichlet(np.ones(len(s_pv))),5)
            rwp1 = alpha_avg(lmp1,r_alpha)
            rwp2 = alpha_avg(lmp2,r_alpha)       
            rt_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            rt_models[i][0].load_state_dict(rwp1)
            rt_models[i][1].load_state_dict(rwp2)
            
            # heuristic qty
            # h1_alpha = np.round(np.array(data_qty)[s_pv]/max(data_qty),5)
            # h1_alpha /= sum(h1_alpha)
            h1_alpha = np.round(np.array(data_qty)[s_pv]/sum(np.array(data_qty)[s_pv]),5)
            h1wp1 = alpha_avg(lmp1,h1_alpha)
            h1wp2 = alpha_avg(lmp2,h1_alpha)
            h1_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            h1_models[i][0].load_state_dict(h1wp1)
            h1_models[i][1].load_state_dict(h1wp2)
            
            # heuristic - uniform ratios
            h2_alpha = 1/len(s_pv) * np.ones(len(s_pv))
            h2wp1 = alpha_avg(lmp1,h2_alpha)
            h2wp2 = alpha_avg(lmp2,h2_alpha)   
            h2_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            h2_models[i][0].load_state_dict(h2wp1)
            h2_models[i][1].load_state_dict(h2wp2)
            
            if oargs.dset_split < 2:
                target_accs[i],_ = test_img_gr(target_models[i][0],target_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)   
                rt_accs[i],_ = test_img_gr(rt_models[i][0],rt_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device) 
                h1_accs[i],_ = test_img_gr(h1_models[i][0],h1_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)    
                h2_accs[i],_ = test_img_gr(h2_models[i][0],h2_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device) 
            elif oargs.dset_split == 2:
                target_accs[i],_ = test_img_gr(target_models[i][0],target_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)   
                rt_accs[i],_ = test_img_gr(rt_models[i][0],rt_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)  
                h1_accs[i],_ = test_img_gr(h1_models[i][0],h1_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)      
                h2_accs[i],_ = test_img_gr(h2_models[i][0],h2_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
        else:
            wap = alpha_avg(lmp,ovr_alpha[:,i])
            wap_dict[i] = wap
            # test the resulting models
            target_models[i] = deepcopy(start_net)
            target_models[i].load_state_dict(wap)
            
            # build random models + test them
            r_alpha = np.round(np.random.dirichlet(np.ones(len(s_pv))),5)
            rwp = alpha_avg(lmp,r_alpha)
            rt_models[i] = deepcopy(start_net)
            rt_models[i].load_state_dict(rwp)
            
            # heuristic qty
            # h1_alpha = np.round(np.array(data_qty)[s_pv]/max(data_qty),5)
            # h1_alpha /= sum(h1_alpha)
            h1_alpha = np.round(np.array(data_qty)[s_pv]/sum(np.array(data_qty)[s_pv]),5)
            h1wp = alpha_avg(lmp,h1_alpha)
            h1_models[i] = deepcopy(start_net)
            h1_models[i].load_state_dict(h1wp)
            
            # heuristic - uniform ratios
            h2_alpha = 1/len(s_pv) * np.ones(len(s_pv))
            h2wp = alpha_avg(lmp,h2_alpha)
            h2_models[i] = deepcopy(start_net)
            h2_models[i].load_state_dict(h2wp)
            
            if oargs.dset_split < 2:
                target_accs[i],_ = test_img_ttest(target_models[i],\
                            oargs.div_bs,d_train,d_dsets[i],device=device)
                rt_accs[i],_ = test_img_ttest(rt_models[i],\
                            oargs.div_bs,d_train,d_dsets[i],device=device)
                h1_accs[i],_ = test_img_ttest(h1_models[i],\
                            oargs.div_bs,d_train,d_dsets[i],device=device)        
                h2_accs[i],_ = test_img_ttest(h2_models[i],\
                            oargs.div_bs,d_train,d_dsets[i],device=device)          
            elif oargs.dset_split == 2:
                target_accs[i],_ = test_img_ttest(target_models[i],\
                            oargs.div_bs,d_train_dict[i],d_dsets[i],device=device)
                rt_accs[i],_ = test_img_ttest(rt_models[i],\
                            oargs.div_bs,d_train_dict[i],d_dsets[i],device=device)
                h1_accs[i],_ = test_img_ttest(h1_models[i],\
                            oargs.div_bs,d_train_dict[i],d_dsets[i],device=device)        
                h2_accs[i],_ = test_img_ttest(h2_models[i],\
                            oargs.div_bs,d_train_dict[i],d_dsets[i],device=device)         
        ## compute energies
        tmp_c2d_rates = d2d_tx_rates[:,i]
        our_nrg += mt_nrg_calc(ovr_alpha[:,i],tmp_c2d_rates)
        r_nrg += mt_nrg_calc(r_alpha,tmp_c2d_rates)
        h1_nrg += mt_nrg_calc(h1_alpha,tmp_c2d_rates)
        h2_nrg += mt_nrg_calc(h2_alpha,tmp_c2d_rates)
    else:
        if oargs.grad_rev == True :
            source_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            source_models[i][0].load_state_dict(lmp[i][0])
            source_models[i][1].load_state_dict(lmp[i][1])
            if oargs.dset_split < 2: 
                source_accs[i],_ = test_img_gr(source_models[i][0],source_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                source_accs[i],_ = test_img_gr(source_models[i][0],source_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
        else:
            source_models[i] = deepcopy(start_net)
            source_models[i].load_state_dict(lmp[i])
            if oargs.dset_split < 2: 
                source_accs[i],_ = test_img_ttest(source_models[i],\
                                oargs.div_bs,d_train,d_dsets[i],device=device)
            elif oargs.dset_split == 2:
                source_accs[i],_ = test_img_ttest(source_models[i],\
                                oargs.div_bs,d_train_dict[i],d_dsets[i],device=device)

## mixed alpha and psi selection - here for ease
oo_models = {} #single source to single target
sm_models = {} #single source to multi-target
oo_nrg = 0
sm_nrg = 0

occupied_sources = np.zeros_like(np.where(np.array(psi_vals)==0)[0])
oo_alpha = deepcopy(ovr_alpha)
avg_odeg = calc_odeg()
sm_alpha = calc_sm_alphas(avg_odeg)

for i,j in enumerate(psi_vals):
    if j == 1:
        t_ind = np.random.randint(0,oargs.l_devices)
        while occupied_sources[t_ind] == 1:
            # if all sources have a match, then reset the vector
            if (occupied_sources == np.ones_like(occupied_sources)).all():
                occupied_sources = np.zeros_like(occupied_sources) 
                t_ind = np.random.randint(0,oargs.l_devices)
            else:
                t_ind = np.random.randint(0,oargs.l_devices)
        occupied_sources[t_ind] = 1 
        oo_alpha2 = np.zeros_like(ovr_alpha[:,i])
        oo_alpha2[t_ind] = 1
        
        if oargs.grad_rev == True:
            oo_wp1 = alpha_avg(lmp1,oo_alpha2)
            oo_wp2 = alpha_avg(lmp2,oo_alpha2)
            oo_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            oo_models[i][0].load_state_dict(oo_wp1)
            oo_models[i][1].load_state_dict(oo_wp2)
            
            sm_wp1 = alpha_avg(lmp1,sm_alpha[:,i])
            sm_wp2 = alpha_avg(lmp2,sm_alpha[:,i])            
            sm_models[i] = [deepcopy(feature_net_base),deepcopy(features_2_class_base)]
            sm_models[i][0].load_state_dict(sm_wp1)       
            sm_models[i][1].load_state_dict(sm_wp2)

            if oargs.dset_split < 2: 
                oo_accs[i],_ = test_img_gr(oo_models[i][0],oo_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)
                sm_accs[i],_ = test_img_gr(sm_models[i][0],sm_models[i][1],\
                            oargs.div_bs,d_train,indx=d_dsets[i],device=device)        
            elif oargs.dset_split == 2:
                oo_accs[i],_ = test_img_gr(oo_models[i][0],oo_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)
                sm_accs[i],_ = test_img_gr(sm_models[i][0],sm_models[i][1],\
                            oargs.div_bs,d_train_dict[i],indx=d_dsets[i],device=device)     
        else:
            oo_wp = alpha_avg(lmp,oo_alpha2)
            oo_models[i] = deepcopy(start_net)
            oo_models[i].load_state_dict(oo_wp)
            
            # one-to-many [approximate avg out degree]
            sm_wp = alpha_avg(lmp,sm_alpha[:,i])
            sm_models[i] = deepcopy(start_net)
            sm_models[i].load_state_dict(sm_wp)
            
            if oargs.dset_split < 2: 
                oo_accs[i],_ = test_img_ttest(oo_models[i],oargs.div_bs,\
                                d_train,d_dsets[i],device=device)
                sm_accs[i],_ = test_img_ttest(sm_models[i],oargs.div_bs,\
                                d_train,d_dsets[i],device=device)            
            elif oargs.dset_split == 2:
                oo_accs[i],_ = test_img_ttest(oo_models[i],oargs.div_bs,\
                                d_train_dict[i],d_dsets[i],device=device)
                sm_accs[i],_ = test_img_ttest(sm_models[i],oargs.div_bs,\
                                d_train_dict[i],d_dsets[i],device=device)              
        
        oo_nrg += mt_nrg_calc(oo_alpha2,tmp_c2d_rates)
        sm_nrg += mt_nrg_calc(sm_alpha[:,i],tmp_c2d_rates)

# %% save the results
import pandas as pd
acc_df = pd.DataFrame()
acc_df['ours'] = list(target_accs.values()) 
acc_df['rng'] = list(rt_accs.values()) 
acc_df['max_qty'] = list(h1_accs.values()) # this is standard FL now
acc_df['unif_ratio'] = list(h2_accs.values())
acc_df['o2o'] = list(oo_accs.values()) 
acc_df['o2m'] = list(sm_accs.values())
# acc_df['source'] = list(source_accs.values())

nrg_df = pd.DataFrame()
nrg_df['ours'] = [our_nrg]
nrg_df['rng'] = [r_nrg]
nrg_df['max_qty'] = [h1_nrg]
nrg_df['unif_ratio'] = [h2_nrg]
nrg_df['o2o'] = [oo_nrg]
nrg_df['o2m'] = [sm_nrg]

print(acc_df)
print(nrg_df)
# if oargs.nrg_mt == 0:
#     if oargs.dset_split == 0: # only one dataset
#         acc_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/seed_'+str(oargs.seed) \
#                 +'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+'_acc.csv')
#         nrg_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/seed_'+str(oargs.seed)\
#                 +'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+'_nrg.csv')
#     else:
#         acc_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/seed_'+str(oargs.seed)\
#                 +'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+'_acc.csv')
#         nrg_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/seed_'+str(oargs.seed)\
#                 +'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+'_nrg.csv')

# else: ## adjust file name with nrg
#     if oargs.dset_split == 0: # only one dataset
#         acc_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/NRG'+str(oargs.phi_e)+'_'\
#                   +'seed_'+str(oargs.seed)+'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+end+end2+'_acc.csv')
#         nrg_df.to_csv(cwd+'/mt_results/'+oargs.dset_type+'/NRG'+str(oargs.phi_e)+'_'\
#                   +'seed_'+str(oargs.seed)+'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+end+end2+'_nrg.csv')                             
#     else:
#         acc_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/NRG'+str(oargs.phi_e)+'_'\
#                   +pre+'seed_'+str(oargs.seed)+'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+end+end2+'_acc.csv')
#         nrg_df.to_csv(cwd+'/mt_results/'+oargs.split_type+'/NRG'+str(oargs.phi_e)+'_'\
#                   +pre+'seed_'+str(oargs.seed)+'_'+oargs.labels_type \
#                   +'_'+oargs.div_nn+end+end2+'_nrg.csv') 





