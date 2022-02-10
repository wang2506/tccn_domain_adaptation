# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 14:43:47 2022

@author: ch5b2
"""
import os
import numpy as np
import pickle as pk
import random
from copy import deepcopy
import scipy 
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

from optim_utils.optim_parser import optim_parser
from div_utils.neural_nets import MLP,CNN,segmentdataset,\
    LocalUpdate, wAvg, test_img

cwd = os.getcwd()
args = optim_parser()
seed = args.seed
np.random.seed(seed)
random.seed(seed)

# %% imports + redefinitions
# comp method
if args.div_comp == 'gpu':
    device = torch.device('cuda:'+str(args.div_gpu_num))
elif args.div_comp == 'cpu':
    device = torch.device('cpu')
else:
    raise TypeError('wrong computation method')

# data qty
with open(cwd+'/data_div/devices'+str(args.t_devices)\
          +'_seed'+str(args.seed)+'_data_qty','rb') as f:
    alld_qty = pk.load(f)

# dataset determination
pwd = os.path.dirname(cwd)
if args.dset_split == 0:
    if args.dset_type == 'M':
        print('Using MNIST \n')
        d_train = torchvision.datasets.MNIST(pwd+'/data/',train=True,download=True,\
                        transform=transforms.ToTensor())
    elif args.dset_type == 'S': #needs scipy
        print('Using SVHN')
        d_train = torchvision.datasets.SVHN(pwd+'/data/svhn/',split='train',download=True,\
                        transform=transforms.ToTensor())
        #http://ufldl.stanford.edu/housenumbers/
        # TODO : need some data-preprocessing
    elif args.dset_type == 'U':
        print('Using USPS')
        d_train = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                        transform=transforms.ToTensor())
    else:
        raise TypeError('Dataset exceeds sims')

elif args.dset_split == 1: # TODO
    if args.dset_type == 'M+S':
        print('Using MNIST + SVHN')
    elif args.dset_type == 'M+U':
        print('Using MNIST + USPS')
    elif args.dset_type == 'S+U':
        print('Using SVHN + USPS')
    elif args.dset_type == 'A':
        print('Using MNIST + SVHN + USPS')
    else:
        raise TypeError('Datasets exceed sims')

# labels assignment
labels = 10 #all three datasets have three labels
if args.label_split == 1:
    # determination of the device datasets requires
    if args.labels_type == 'mild':
        lpd = [sorted(random.sample(range(labels),3)) for i in range(args.t_devices)]
    elif args.labels_type == 'extreme':
        lpd = [random.sample(range(labels),1) for i in range(args.t_devices)]
    else:
        raise TypeError('labels type invalid')
elif args.label_split == 0: #i.e., iid
    lpd = [list(range(10)) for i in range(args.t_devices)]
else:
    raise TypeError('label split invalid')

# lpd[1] = [1]
# lpd[0] = [0]
lpd[0] = lpd[1]

# populate device datasets
d_dsets = {} #these are indexes
d_train_ls = {i:np.where(d_train.targets==i)[0] for i in range(labels)}

d_dset_sqtys = [np.random.multinomial(alld_qty[i],[0.333]*3) \
                         for i in range(args.t_devices)]

# d_dset_sqtys[1] = [d_dset_sqtys[1][0]]
# d_dset_sqtys[0] = [d_dset_sqtys[0][0]]

for i in range(args.t_devices): 
    d_dsets[i] = []
    c_labels = lpd[i]
    for ti,tj in enumerate(d_dset_sqtys[i]):
        td_dset = random.sample(d_train_ls[c_labels[ti]].tolist(),tj)
        d_dsets[i].extend(td_dset)

# %% source target label re-assignment func
def st_relab(s_dset,t_dset,d_train):
    ## populate the set
    sl_dset = [(d_train[td][0],0) for td in s_dset]
    tl_dset = [(d_train[td][0],1) for td in t_dset]
    return sl_dset, tl_dset

# %% single training iter func
def div_roi(loc_model,bs,lr,l_dset=None,st=None,\
            device=device,dt=d_train): #divergence_run_one_iteration
    tobj = LocalUpdate(device,bs=bs,lr=lr,epochs=1,st=st,\
            dataset=dt,indexes=l_dset)
    _,w,loss = tobj.train(net=loc_model.to(device))
    return w,loss

# %% setup training vars
if args.div_nn == 'MLP':
    d_in = np.prod(d_train[0][0].shape)
    d_h = 64
    d_out = 2
    start_net = MLP(d_in,d_h,d_out).to(device)

    try:
        with open(cwd+'/div_utils/MLP_start_w','rb') as f:
            start_w = pk.load(f)
        start_net.load_state_dict(start_w)
    except:
        start_w = start_net.state_dict()
        with open(cwd+'/div_utils/MLP_start_w','wb') as f:
            pk.dump(start_w,f)
elif args.div_nn == 'CNN':
    nchannels = 1
    nclasses = 2
    start_net = CNN(nchannels,nclasses).to(device)

    try:
        with open(cwd+'/div_utils/CNN_start_w','rb') as f:
            start_w = pk.load(f)
        start_net.load_state_dict(start_w)
    except:
        start_w = start_net.state_dict()
        with open(cwd+'/div_utils/CNN_start_w','wb') as f:
            pk.dump(start_w,f)

print(start_net)

# %% pairwise training loops
# lab2ulab_accs = np.zeros(shape=(args.l_devices,args.t_devices)).astype(int) #labeled to unlabeled accuracies
lab2ulab_accs = np.zeros(shape=(args.t_devices,args.t_devices)).astype(int) #labeled to unlabeled accuracies

# for i in range(args.l_devices): #a device with labeled data
for i in range(args.t_devices):
    for j in range(args.t_devices):
        if lab2ulab_accs[i,j] == 0 and i != j:
            # get the relabelled datasets
            sl_dset,tl_dset = st_relab(d_dsets[i],d_dsets[j],d_train)
            
            st_net = deepcopy(start_net)
            
            # training, combining, and testing loop
            for tc in range(args.div_ttime):
                # one training iteration
                s_temp_set = random.sample(d_dsets[i],10*args.div_bs)
                s_w,s_loss = div_roi(deepcopy(st_net),st='source',\
                        bs=args.div_bs,lr=args.div_lr,\
                        l_dset=s_temp_set)#random.sample(d_dsets[i],args.div_bs))
                #d_dsets[i])
                t_temp_set = random.sample(d_dsets[j],10*args.div_bs) 
                t_w,t_loss = div_roi(deepcopy(st_net),st='target',\
                        bs=args.div_bs,lr=args.div_lr,\
                        l_dset=t_temp_set)#random.sample(d_dsets[j],args.div_bs))
                #d_dsets[j])
                # perform unweighted avg for the two devices
                w_avg = wAvg([s_w,t_w])
                st_net.load_state_dict(w_avg)
                
                # print(s_loss)
                # print(t_loss)
                # print(w_avg['layer_hidden.bias'])
                
            # calc acc/error - done in distributed way
            s_acc,s_loss = test_img(st_net,args.div_bs,dset=d_train,\
                     indx=random.sample(d_dsets[i],10*args.div_bs),\
                     st='source',device=device)
            
            t_acc,t_loss = test_img(st_net,args.div_bs,dset=d_train,\
                     indx=random.sample(d_dsets[j],10*args.div_bs),\
                     st='target',device=device)
            
            ovr_acc = (s_acc+t_acc)/2
            lab2ulab_accs[i,j] = ovr_acc
            try:
                lab2ulab_accs[j,i] = ovr_acc
            except:
                fill_var = 1

print('done')

# %% save the results
with open(cwd+'/div_results/test_ex','wb') as f:
    pk.dump(lab2ulab_accs,f)



# %% divergence calc

# def div_calc():
    
#     return True


# if __name__ == "__main__":
#     bval = div_calc() 
#     print(div_calc())