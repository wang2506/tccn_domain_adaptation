# -*- coding: utf-8 -*-
"""
@author: ch5b2
"""

import os
import cvxpy as cp
import numpy as np
import pickle as pk
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset


from optim_utils.optim_parser import optim_parser
from div_utils.neural_nets import init_source_train, MLP, CNN, test_img_strain, GCNN
from mnist_m import MNISTM

cwd = os.getcwd()
args = optim_parser()

np.random.seed(args.seed)
random.seed(args.seed)

if args.label_split == 0:
    args.labels_type = 'iid'

# %% variable declarations
phi_s = args.phi_s # source errors
phi_t = args.phi_t # target errors
phi_e = args.phi_e # energy consumptions

psi = cp.Variable(args.t_devices,pos=True)#,boolean=True)
alpha = cp.Variable((args.t_devices,args.t_devices),pos=True)

# %% shared entities
## data qty
all_u_qtys = np.random.normal(args.avg_uqty,args.avg_uqty/6,\
            size=args.u_devices).astype(int) #all_unlabelled_qtys

split_lqtys = np.random.normal(args.avg_lqty_l,args.avg_lqty_l/6,\
            size=args.l_devices).astype(int)
split_uqtys = np.random.normal(args.avg_lqty_u,args.avg_lqty_u/6,\
            size=args.l_devices).astype(int)

print(all_u_qtys)
print(split_lqtys)
print(split_uqtys)

net_l_qtys = split_lqtys + split_uqtys

data_qty_alld = list(net_l_qtys)+list(all_u_qtys)

with open(cwd+'/data_div/devices'+str(args.t_devices)+\
          '_seed'+str(args.seed)+'_data_qty','wb') as f:
    pk.dump(data_qty_alld,f)

## epsilon hat - empirical loss at as measured on s_devices' labelled data
# full org datasets
pwd = os.path.dirname(cwd)
if args.dset_split == 0:
    if args.dset_type == 'M':
        print('Using MNIST \n')
        d_train = torchvision.datasets.MNIST(pwd+'/data/',train=True,download=True,\
                        transform=transforms.ToTensor())
    elif args.dset_type == 'S': #needs scipy
        print('Using SVHN \n')
        args.approx_iters = 100
        tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),\
                        transforms.Grayscale(),transforms.CenterCrop(28)])
        d_train = torchvision.datasets.SVHN(pwd+'/data/svhn/',split='train',download=True,\
                        transform=tx_dat)
        d_train.targets = d_train.labels
        #http://ufldl.stanford.edu/housenumbers/
    elif args.dset_type == 'U':
        print('Using USPS \n')
        tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])
        try: 
            d_train = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                            transform=tx_dat)
        except:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context            
            d_train = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                            transform=tx_dat)
        d_train.targets = np.array(d_train.targets)
    elif args.dset_type == 'MM':
        print('Using MNIST-M \n')
        tx_dat =  torchvision.transforms.Compose([transforms.ToTensor()])
        d_train = MNISTM(pwd+'/data/',train=True,download=True,\
                         transform=tx_dat)        
    else:
        raise TypeError('Dataset exceeds sims')
elif args.dset_split == 1:
    tx_m = torchvision.transforms.Compose([transforms.ToTensor()])
    tx_mm = torchvision.transforms.Compose([transforms.ToTensor(),\
                transforms.Grayscale()])
    tx_u = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])    
    
    d_m = torchvision.datasets.MNIST(pwd+'/data/',train=True,download=True,\
                    transform=tx_m)
    d_mm = MNISTM(pwd+'/data/',train=True,download=True,\
                     transform=tx_mm)        
    try: 
        d_u = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                        transform=tx_u)
    except:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context            
        d_u = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                        transform=tx_u)
    d_u.targets = torch.tensor(d_u.targets)    
    
    if args.split_type == 'M+MM':
        print('Using MNIST + MNIST-M')
        d_train = d_m+d_mm
        d_train.targets = torch.concat([d_m.targets,d_mm.targets])
    elif args.split_type == 'M+U':
        print('Using MNIST + USPS')
        d_train = d_m+d_u
        d_train.targets = torch.concat([d_m.targets,d_u.targets])
    elif args.split_type == 'MM+U':
        print('Using MNIST-M + USPS')
        d_train = d_mm+d_u       
        d_train.targets = torch.concat([d_mm.targets,d_u.targets])
    elif args.split_type == 'A':
        print('Using MNIST + MNIST-M + USPS')
        d_train = d_m+d_mm+d_u
        d_train.targets = torch.concat([d_m.targets,d_mm.targets,d_u.targets])
    else:
        raise TypeError('Datasets exceed sims')

# data processing
if args.dset_split == 0:
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.dset_type+'_'+args.labels_type+'_lpd','rb') as f:
        lpd = pk.load(f)
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.dset_type+'_'+args.labels_type+'_dindexsets','rb') as f:
        d_dsets = pk.load(f)    
else:
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.split_type+'_'+args.labels_type+'_lpd','rb') as f:
        lpd = pk.load(f)
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.split_type+'_'+args.labels_type+'_dindexsets','rb') as f:
        d_dsets = pk.load(f)    

# random sampling to determine the labelled datasets
ld_sets = {}
for i in d_dsets.keys():
    if i >= args.l_devices:
        break
    else:
        ld_sets[i] = random.sample(d_dsets[i],split_lqtys[i])

if args.init_test != 1:
    ## call source training here
    # setup training vars
    if args.div_comp == 'gpu':
        device = torch.device('cuda:'+str(args.div_gpu_num))
    else:
        device = torch.device('cpu')
    if args.div_nn == 'MLP':
        d_in = np.prod(d_train[0][0].shape)
        d_h = 64
        d_out = 10
        start_net = MLP(d_in,d_h,d_out).to(device)
    
        try:
            with open(cwd+'/optim_utils/MLP_start_w','rb') as f:
                start_w = pk.load(f)
            start_net.load_state_dict(start_w)
        except:
            start_w = start_net.state_dict()
            with open(cwd+'/optim_utils/MLP_start_w','wb') as f:
                pk.dump(start_w,f)
    elif args.div_nn == 'CNN':
        nchannels = 1
        nclasses = 10
        start_net = CNN(nchannels,nclasses).to(device)
        try:
            with open(cwd+'/optim_utils/CNN_start_w','rb') as f:
                start_w = pk.load(f)
            start_net.load_state_dict(start_w)
        except:
            start_w = start_net.state_dict()
            with open(cwd+'/optim_utils/CNN_start_w','wb') as f:
                pk.dump(start_w,f)
    else:
        nchannels = 1
        nclasses = 10
        start_net = GCNN(nchannels,nclasses).to(device)    
    
    # # sequential storage
    hat_ep = []
    hat_w = {}
    ld_nets = [deepcopy(start_net) for i in range(args.l_devices)]
    print('training source domains - find source errors')
    for i in range(args.l_devices):
        # start_net.load_state_dict(start_w)
        # train the source model on labeled data
        params_w,ce_loss_t = init_source_train(ld_sets[i],args=args,\
                d_train=d_train,nnet=deepcopy(ld_nets[i]),device=device)
        # obtain the source accuracy on the full local dataset
        t_net = deepcopy(ld_nets[i])
        t_net.load_state_dict(params_w)
        acc_i,ce_loss = test_img_strain(t_net,\
                    args.div_bs,d_train,indx=d_dsets[i],device=device)
            
        hat_ep.append((100-acc_i)/100) # need to replace with the final training error
        # print(acc_i)
        hat_w[i] = params_w
    
    if args.dset_split == 0:
        with open(cwd+'/source_errors/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.dset_type+'_'+args.labels_type+'_modelparams_'+args.div_nn,\
            'wb') as f:
            pk.dump(hat_w,f)
    else:
        with open(cwd+'/source_errors/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.split_type+'_'+args.labels_type+'_modelparams_'+args.div_nn,\
            'wb') as f:
            pk.dump(hat_w,f)     
else:
    # temporarily assign constant values
    # later, we will need to figure out a way to estimate them
    # hat_ep = []
    # min_ep_vect = 1e-3
    # max_ep_vect = 5e-1
    # temp_ep_vect = (min_ep_vect+(max_ep_vect-min_ep_vect) \
    #                 *np.random.rand(args.t_devices)).tolist()
    # for i in range(args.l_devices):
    #     t_hat_ep = random.sample(temp_ep_vect,1)
    #     hat_ep.extend(t_hat_ep)
    hat_ep = [0.04,0.27,0.45,0.09,0.45] 

## ordering devices (labelled and unlabelled combined)
# for now, just sequentially, all labelled, then unlabelled
device_order = list(np.arange(0,args.l_devices+args.u_devices,1))

hat_ep_alld = []
for i in range(args.t_devices):
    if i < args.l_devices:
        # hat_ep[2] = 80
        hat_ep_alld.append(hat_ep[i]) #*1e3)
    else: 
        # temp_factor = np.round(np.log(max(hat_ep)))+6
        # hat_ep_alld.append(np.power(10,temp_factor))
        # hat_ep_alld.append(1e4) #1e3
        hat_ep_alld.append(1e3)

## empirical hypothesis mismatch error
# can't really be done in practice - randomly assignment aar
ep_mismatch = {}
min_mismatch = 1e-3
max_mismatch = 5e-1

for i in range(args.t_devices):
    temp_ep_mismatch = (min_mismatch+(max_mismatch-min_mismatch)\
                        *np.random.rand(args.t_devices)).tolist()
    ep_mismatch[i] = []
    for j in range(args.t_devices):
        ep_mismatch[i].extend(random.sample(temp_ep_mismatch,1))

## divergence terms
if args.div_flag == 1: #div flag is online
    if args.dset_split == 0: 
        with open(cwd+'/div_results/div_vals/devices'+str(args.t_devices)\
            +'_seed'+str(args.seed)+'_'+args.div_nn\
            +'_'+args.dset_type\
            +'_'+args.labels_type,'rb') as f:
            div_pairs = pk.load(f)
    else:
        with open(cwd+'/div_results/div_vals/devices'+str(args.t_devices)\
            +'_seed'+str(args.seed)+'_'+args.div_nn\
            +'_'+args.split_type\
            +'_'+args.labels_type,'rb') as f:
            div_pairs = pk.load(f)
else:
    div_pairs = np.ones((args.t_devices,args.t_devices))

# divergence normalization (by low end - 50)
d_min = 50
d_max = 100
for ir,row in enumerate(div_pairs):
    for iv,cval in enumerate(row):
        if cval != 0:
            cval = (cval-50)*2
            row[iv] = cval
            div_pairs[ir] = row

## rademacher estimates
rad_s = [np.sqrt(2*np.log(net_l_qtys[i])/net_l_qtys[i]) for i in range(args.l_devices)]
rad_t = [np.sqrt(2*np.log(all_u_qtys[i])/all_u_qtys[i]) for i in range(args.u_devices)]

rad_alld = rad_s+rad_t

## sqrt of log term
sqrt_s = [3*np.sqrt(np.log(2/args.l_delta)/(2*net_l_qtys[i])) for i in range(args.l_devices)]
sqrt_t = [3*np.sqrt(np.log(2/args.l_delta)/(2*all_u_qtys[i])) for i in range(args.u_devices)]

sqrt_alld = sqrt_s+sqrt_t

# %% objective fxn - term 1 and 2 [source and target errors]
## fxn name --> posy_err_calc
def err_calc(psi,chi,chi_init,psi_init,err_type,alpha_init=None,div_flag=False,\
               rads=rad_alld,sqrts=sqrt_alld,hat_ep=hat_ep_alld,\
                ep_mis=ep_mismatch,div_vals=None,args=args):
    err_denoms = []
    if err_type == 's':
        chi_scale = np.array(hat_ep) + 2*np.array(rads) + np.array(sqrts)
        
        chi_init = np.divide(chi_init,chi_scale)
        chi_var_scale = [chi[itsc]/tcs for itsc,tcs in enumerate(chi_scale)] #np.divide(chi,chi_scale)
        ovr_init = psi_init + chi_init
        
        for i in range(args.t_devices):
            err_denom = cp.power(psi[i]*ovr_init[i]/psi_init[i],\
                                   psi_init[i]/ovr_init[i])
            err_denom *= cp.power(chi_var_scale[i]*ovr_init[i]/chi_init[i], \
                                    chi_init[i]/ovr_init[i])
            err_denoms.append(err_denom)
            
        return err_denoms,chi_scale #chi_scale for debug
    
    elif err_type == 't':
        chi_scale = {} # need a unique scaling for each i,j pair
        chi_scale_init = {} # for alpha_init
        err_denoms = {}
        # chi_init_lists = []
        for j in range(args.t_devices):
            chi_scale[j] = []
            chi_scale_init[j] = []
            for i in range(args.t_devices):
                if div_flag == False:
                    cs_factor = hat_ep[i]+2*rad_alld[i]+sqrts[i]\
                                +4*rad_alld[j]+sqrts[j] #+ep_mis[j][i]
                else:
                    cs_factor = hat_ep[i]+2*rad_alld[i]+sqrts[i]\
                                +4*rad_alld[j]+sqrts[j]+\
                                0.5*2*div_vals[i,j]/100+\
                                2*(rad_alld[i]+rad_alld[j]) + \
                                sqrts[i]+sqrts[j] #+ep_mis[j][i]
                                                                
                    # by defn, divergence is 2*(1-min error)
                    # equiv 2*(accuracy/100), and its scaled by 1/2 in our obj fxn
                    
                # print(cs_factor)
                
                t_chi_scale = alpha[i,j] * cs_factor
                t_chi_scale_init = alpha_init[i,j] * cs_factor
                
                chi_scale[j].append(t_chi_scale)     
                chi_scale_init[j].append(t_chi_scale_init)
            chi_scale[j] = np.array(chi_scale[j])
            chi_scale_init[j] = np.array(chi_scale_init[j])
            
            temp_chi_init = np.divide(chi_init[:][j],chi_scale_init[j])
            # chi[:,j] = cp.multiply(chi[:,j],cp.power(cp.hstack(chi_scale[j]),-1))
            temp_chi = cp.multiply(chi[:,j],cp.power(cp.hstack(chi_scale[j]),-1))
            ovr_init = psi_init + temp_chi_init #chi_init[:,j]
            
            err_denoms[j] = []
            for i in range(args.t_devices):
                err_denom = cp.power(psi[i]*ovr_init[i]/psi_init[i],\
                                     psi_init[i]/ovr_init[i])
                err_denom *= cp.power(temp_chi[i]*ovr_init[i]/temp_chi_init[i],\
                                     temp_chi_init[i]/ovr_init[i])
                err_denoms[j].append(err_denom)
            
        return err_denoms,chi_scale,chi_scale_init
    else:
        raise TypeError('Wrong error type')

## auxiliary variables
chi_s = cp.Variable(args.t_devices,pos=True) 
chi_t = cp.Variable((args.t_devices,args.t_devices),pos=True)

# %% constraints
constraints = []

# auxiliary vars for these two constraints
chi_c1 = cp.Variable(pos=True) #args.t_devices,pos=True)
chi_c2 = cp.Variable(pos=True) #(args.t_devices,args.t_devices),pos=True)
# chi_c3 = cp.Variable(pos=True)

# alpha constraints
con_alpha = []
for i in range(args.t_devices):
    for j in range(args.t_devices):
        con_alpha.append(alpha[i,j] <= 1)
        # con_alpha.append(alpha[i,j] >= 1e-3) #1e-3) #1e-6)
        # con_alpha.append(psi[i]*alpha[i,j] <= 1e-3) #1e-2) #1e-3)
        con_alpha.append(alpha[i,j] >= 1e-3) #1e-3) #1e-6)
        con_alpha.append(psi[i]*alpha[i,j] <= 1e-3) #chi_c3) #1e-3)
    con_alpha.append(cp.sum(alpha[:,i]) <= 1+1e-6)
    # con_alpha.append(psi[i]*cp.sum(alpha[:,i]) <= 1+1e-6)
constraints.extend(con_alpha)

# psi constraints
con_psi = []
for i in range(args.t_devices):
    con_psi.append(psi[i] <= 1+1e-6)
    con_psi.append(psi[i] >= 1e-6)
constraints.extend(con_psi)

# cent_epsilon = 1e-2 #1e-1 #5e-2 #5e-2 #1e-1 #5e-2 #5e-2 #4e-2 #7e-3 #1e-2 #5e-3
cent_epsilon = 1e-2

con_prev = []
con_prev.append(chi_c1 <= 1e-4) #1e-3)#1e-6)
con_prev.append(chi_c1 >= 1e-8)
# for j in range(args.t_devices):
#     # con_prev.append(chi_c1 <= 1e-1)
#     # for i in range(args.t_devices):
#     #     # con_prev.append(chi_c2 <= 1e-3) #1e-3)#1e-6)
#     #     con_prev.append(chi_c2 <= 1e-4)
#     #     con_prev.append(chi_c2 >= 1e-8)

constraints.extend(con_prev)

def con_posy_denom_calc(chi,chi_init,psi,psi_init,alpha,alpha_init,cp_type,\
                args=args,cp_epsilon=cent_epsilon):    
    if cp_type == 1: #first constraint, return both pos and neg denoms
        pos_t_con_denoms = []
        neg_t_con_denoms = []
        
        for j in range(args.t_devices):
            cur_init_pos = chi_init + psi_init[j] + cp_epsilon
            pos_t1_denom = cp.power(chi*cur_init_pos/chi_init, \
                                  chi_init/cur_init_pos)
            pos_t2_denom = cp.power(psi[j]*cur_init_pos/psi_init[j], \
                                  psi_init[j]/cur_init_pos)
            pos_t3_denom = cp.power(cur_init_pos, \
                                  cp_epsilon/cur_init_pos)
            pos_t_con_denoms.append(pos_t1_denom*pos_t2_denom*pos_t3_denom)            
            
            cur_init_neg = np.sum(alpha_init[:,j])+cp_epsilon
            neg_t_hold = cp.power(cur_init_neg, \
                                 cp_epsilon/cur_init_neg)
            for i in range(args.t_devices):
                neg_t_hold *= cp.power(alpha[i][j]*cur_init_neg/alpha_init[i][j],\
                            alpha_init[i][j]/cur_init_neg)
            neg_t_con_denoms.append(neg_t_hold)
        
    elif cp_type == 2: #second constraints, return both pos and neg
        pos_t_con_denoms = {}
        neg_t_con_denoms = {}    
        
        # t_pos_epsilon = 2e-2 #1e-2
        
        for i in range(args.t_devices):
            pos_t_con_denoms[i] = []
            neg_t_con_denoms[i] = []
            for j in range(args.t_devices):
                cur_init_pos = chi_init + cp_epsilon + \
                    psi_init[j]*alpha_init[i][j]          
                pos_t1_denom = cp.power(chi*cur_init_pos/chi_init, \
                                 chi_init/cur_init_pos)
                pos_t2_denom = cp.power(psi[j]*alpha[i][j]*cur_init_pos \
                                / (psi_init[j]*alpha_init[i][j]), \
                                 (psi_init[j]*alpha_init[i][j])/cur_init_pos)
                pos_t3_denom = cp.power(cur_init_pos, \
                                  cp_epsilon/cur_init_pos)              
                pos_t_con_denoms[i].append(pos_t1_denom*pos_t2_denom*pos_t3_denom)          
                
                cur_init_neg = (1+psi_init[i])*alpha_init[i][j] + cp_epsilon            
                neg_t1_denom = cp.power(alpha[i][j]*cur_init_neg/alpha_init[i][j],\
                                alpha_init[i][j]/cur_init_neg)
                neg_t2_denom = cp.power(alpha[i][j]*psi[i]*cur_init_neg/ \
                                (alpha_init[i][j]*psi_init[i]),\
                                    alpha_init[i][j]*psi_init[i]/cur_init_neg)
                neg_t3_denom = cp.power(cur_init_neg, \
                                cp_epsilon/cur_init_neg)
                neg_t_con_denoms[i].append(neg_t1_denom*neg_t2_denom*neg_t3_denom)

    else:
        raise TypeError('invalid con_posy cp_type')
    
    return pos_t_con_denoms, neg_t_con_denoms

def build_ts_posy_cons(pos_denoms,neg_denoms,chi,cp_type,\
            alpha,psi,args=args,cp_epsilon=cent_epsilon):
    pos_t_con_prev = []
    neg_t_con_prev = []
    
    if cp_type == 1: #first constraint \sum \alpha - psi = 0
        for j in range(args.t_devices):
            pos_t_con_prev.append( (cp.sum(alpha[:,j])/pos_denoms[j]) <= 1)
            neg_t_con_prev.append( ((psi[j]+chi)/neg_denoms[j]) <= 1)
    elif cp_type == 2: #second constraint (1+psi_i-psi_j)alpha = 0
        for i in range(args.t_devices):
            for j in range(args.t_devices):
                pos_t_con_prev.append(((1+psi[i])*alpha[i][j] \
                                      /pos_denoms[i][j])<=1)
                neg_t_con_prev.append( ((chi \
                                +psi[j]*alpha[i][j]) \
                                /neg_denoms[i][j]) <=1 )
    else:
        raise TypeError('cptype invalid posy con')

    return pos_t_con_prev,neg_t_con_prev

# fxn to repeat build source constraints
def build_posy_cons(denoms,args=args):
    s_list = []
    
    for c_denom in denoms:
        s_list.append(1/c_denom <= 1)
    return s_list


# %% posynomial approximation fxn
def posy_init(iter_num,cp_vars,args=args):
    ## iter_num - iteration number
    ## cp_vars - dict of optimization variables on which to perform posynomial approximation
    if iter_num == 0:
        psi_init = 0.5*np.ones(args.t_devices)
        chi_s_init = 100*np.ones(args.t_devices)
        chi_t_init = (100*np.ones((args.t_devices,args.t_devices))).tolist()
        alpha_init = 1e-2*np.ones((args.t_devices,args.t_devices))
        chi_c1 = 1e-7#*np.ones(args.t_devices)
        chi_c2 = 1e-7#*np.ones((args.t_devices,args.t_devices))
        
    else:
        psi_init = cp_vars['psi'].value
        chi_s_init = cp_vars['chi_s'].value
        chi_t_init = cp_vars['chi_t'].value
        alpha_init = cp_vars['alpha'].value
        chi_c1 = cp_vars['chi_c1'].value
        chi_c2 = cp_vars['chi_c2'].value

    return psi_init,chi_s_init,chi_t_init,alpha_init,\
        chi_c1,chi_c2

# %% calculate energy consumption
def c_nrg_calc(var_dict,t_args,a_init,M=1e6):
    # setup needed variables
    ep_E = 1e-3# if alpha smaller than 1e-3, this should zero things out
    vd = var_dict
    tc_alpha = vd['alpha']
    tc_st = vd['psi'] #temp_nrg_source_target
    param_2_bits = M
    
    # want to do 23dbm (0.2) to 25dbm (0.32)
    tx_powers = 0.2 + (0.32-0.2)*np.random.rand(t_args.t_devices)  
    tx_powers = tx_powers.tolist()
    
    # communications constants
    carrier_freq = 2 * 1e9
    noise_db = 4e-21 #-174 dBm/Hz, we convert to watts
    univ_bandwidth = 2e6 #MHz #10 MHz
    mu_tx = 4*np.pi * carrier_freq/(3*1e8)
    eta_los = 2 #3db 
    eta_nlos = 200 #23db
    path_loss_alpha = 2
    psi_tx = 11.95
    beta_tx = 0.14
    dist_d2d_max = 100 #dist_d2d meters
    dist_d2d_min = 25

    # tx_rates
    d2d_tx_rates = np.zeros(shape=(t_args.t_devices,t_args.t_devices))
    for q in range(t_args.t_devices):
        for j in range(t_args.t_devices):
            dist_qj = dist_d2d_min + (dist_d2d_max-dist_d2d_min) \
                *np.random.rand() # randomly determined
            
            theta_qj = 180/np.pi * np.arcsin(dist_d2d_min / dist_qj )
            
            prob_los = 1/(1+ psi_tx * np.exp(-beta_tx*(theta_qj-psi_tx)) )
            prob_nlos = 1-prob_los
            
            la2g_qj = (mu_tx * dist_qj)**path_loss_alpha *\
                (prob_los*eta_los + prob_nlos * eta_nlos )
            
            d2d_tx_rates[q,j] = univ_bandwidth *\
                np.log2(1 + (tx_powers[q]/la2g_qj) / noise_db )    
    
    # calculate energy used for model transferring
    tmp_nrg = 1e-6 # total energy consumption init
    
    for i in range(t_args.t_devices):
         for j in range(t_args.t_devices):
             # temp rescaled alpha to 0 or 1 (if alpha > alpha_min -> 1, else 0)
             # ts_alpha = tc_alpha[i,j]/(tc_alpha[i,j]+ep_E)
             # rescaled alpha need a posynomial approximation
             rs_alpha_init = ep_E + a_init[i,j]
             ts_alpha = cp.power(tc_alpha[i,j]*rs_alpha_init/(alpha_init[i,j]), \
                    alpha_init[i,j]/rs_alpha_init) \
                 * cp.power(rs_alpha_init,ep_E/rs_alpha_init)            

             tmp_nrg += param_2_bits/d2d_tx_rates[i,j] * tx_powers[i] * ts_alpha
    
    return tmp_nrg

# %% combine and run
obj_vals = []
nrg_vals = []
psi_track = {}

for c_iter in range(args.approx_iters):
    t_dict = {'psi':psi,'chi_s':chi_s,'chi_t':chi_t,'alpha':alpha,\
              'chi_c1':chi_c1,'chi_c2':chi_c2}
    psi_init,chi_s_init,chi_t_init,alpha_init,chi_c1_init,chi_c2_init \
        = posy_init(c_iter,t_dict)
    
    # source error term
    s_denoms,chi_s_scale = err_calc(psi,chi_s,chi_s_init,psi_init,err_type='s') 
    s_posy_con = build_posy_cons(s_denoms)
    s_err = phi_s*cp.sum(chi_s)
    
    # target error term
    if args.div_flag == 0: 
        t_denoms,chi_t_scale,chi_t_scale_init = err_calc(psi,chi_t,chi_t_init,\
                            psi_init,err_type='t',alpha_init=alpha_init)
    else:
        t_denoms,chi_t_scale,chi_t_scale_init = err_calc(psi,chi_t,chi_t_init,\
                            psi_init,err_type='t',alpha_init=alpha_init,\
                            div_flag=True,div_vals=div_pairs)
    
    t_posy_cons = []
    for key_td,list_td in t_denoms.items():
        t_posy_con = build_posy_cons(list_td)
        t_posy_cons.extend(t_posy_con)
    
    for j in range(args.t_devices):
        if j == 0:
            t_err = psi[j]*cp.sum(chi_t[:,j])
        else:
            t_err += psi[j]*cp.sum(chi_t[:,j])
    t_err = phi_t*t_err
    
    # build constraint posynomial updates
    con1_denoms_pos,con1_denoms_neg = \
        con_posy_denom_calc(chi_c1,chi_c1_init,psi,psi_init,\
                            alpha,alpha_init,cp_type=1)
    pos_prev_con1,neg_prev_con1 = \
        build_ts_posy_cons(con1_denoms_pos,con1_denoms_neg,chi_c1,cp_type=1, \
                            alpha=alpha,psi=psi)
    
    posy_con = s_posy_con + t_posy_cons
    net_con = constraints + posy_con 
    net_con += pos_prev_con1 + neg_prev_con1
    
    # TODO 
    e_err = c_nrg_calc(t_dict,args,a_init=alpha_init)
    e_err = phi_e*e_err
    
    obj_fxn = s_err + t_err + e_err
    prob = cp.Problem(cp.Minimize(obj_fxn),constraints=net_con)
    prob.solve(solver=cp.MOSEK,gp=True)#,verbose=True)
    
    # print checking
    print('\n current iteration: '+str(c_iter))
    print('prob value:')
    print(prob.value)
    print('psi:')
    print(psi.value)
    # print('chi_s:')
    # print(chi_s.value)
    # print('chi_t:')
    # print(chi_t.value)
    print('alpha:')
    print([cp.sum(alpha[:,j]).value for j in range(args.t_devices)])  
    # print('chi_c1:')
    # print(chi_c1.value)
    # print('chi_c2:')
    # print(chi_c2.value) #[0,:].value)
    # print('chi_c3:')
    # print(chi_c3.value)
    
    obj_vals.append(prob.value)
    psi_track[c_iter] = psi.value
    nrg_vals.append(e_err.value)

# %% saving some results 
if args.div_flag == 1: 
    if args.dset_split == 0:
        with open(cwd+'/optim_results/obj_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.dset_type+'_'+args.labels_type,'wb') as f:
            pk.dump(obj_vals,f)
        
        with open(cwd+'/optim_results/psi_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.dset_type+'_'+args.labels_type,'wb') as f:
            pk.dump(psi_track,f)
        
        with open(cwd+'/optim_results/hat_ep_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.dset_type+'_'+args.labels_type,'wb') as f:
            pk.dump(hat_ep_alld,f)
        
        with open(cwd+'/optim_results/alpha_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.dset_type+'_'+args.labels_type,'wb') as f:
            pk.dump(alpha.value,f)
    else:
        with open(cwd+'/optim_results/obj_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.split_type+'_'+args.labels_type,'wb') as f:
            pk.dump(obj_vals,f)
        
        with open(cwd+'/optim_results/psi_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.split_type+'_'+args.labels_type,'wb') as f:
            pk.dump(psi_track,f)
        
        with open(cwd+'/optim_results/hat_ep_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.split_type+'_'+args.labels_type,'wb') as f:
            pk.dump(hat_ep_alld,f)
        
        with open(cwd+'/optim_results/alpha_val/NRG_'+str(args.phi_e)+'_'\
            +'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.split_type+'_'+args.labels_type,'wb') as f:
            pk.dump(alpha.value,f)
else: #ablation cases for div_flag == 0
    with open(cwd+'/optim_results/obj_val/bgap_st1','wb') as f:
        pk.dump(obj_vals,f)
    
    with open(cwd+'/optim_results/psi_val/bgap_st1','wb') as f:
        pk.dump(psi_track,f)
    
    with open(cwd+'/optim_results/hat_ep_val/hat_ep1','wb') as f:
        pk.dump(hat_ep_alld,f)
    
    with open(cwd+'/optim_results/alpha_val/alpha1','wb') as f:
        pk.dump(alpha.value,f)


