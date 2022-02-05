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

from optim_utils.optim_parser import optim_parser
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset


cwd = os.getcwd()
args = optim_parser()
seed = args.seed

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
        d_train = torchvision.datasets.MNIST(pwd+'/data/',train=True,download=False,\
                        transform=transforms.ToTensor())
    elif args.dset_type == 'S':
        print('Using SVHN')
        d_train = torchvision.datasets.SVHN(pwd+'/data/',train=True,download=True,\
                        transform=transforms.ToTensor())
    elif args.dset_type == 'U':
        print('Using USPS')
    else:
        raise TypeError('Dataset exceeds sims')
elif args.dset_split == 1:
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





# def div_calc():
    
#     return True


# if __name__ == "__main__":
#     bval = div_calc() 
#     print(div_calc())