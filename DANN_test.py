# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:12:15 2021

@author: ch5b2
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from copy import deepcopy
import random
import pickle

from utils.tf_parser import tf_parser
from utils.neural_nets import DANN_CNN_F, DANN_CNN_C, \
    GradReverse, grad_reverse, DANN_CNN_D, LocalUpdate, segmentdataset
from utils.testing import test_img, test_img2


args = tf_parser()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.comp == 'gpu':
    device = torch.device('cuda:'+args.gpu_num)
else:
    device = torch.device('cpu')


if args.data_style == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(),\
                                      transforms.Normalize((0.1307,),(0.3081,))])
    d_train = torchvision.datasets.MNIST('./data/mnist/',train=True,download=False,\
                                         transform=trans_mnist)
    d_test = torchvision.datasets.MNIST('./data/mnist/',train=False,download=False,\
                                        transform=trans_mnist)
elif args.data_style == 'fmnist':
    d_train = torchvision.datasets.FashionMNIST('./data/fmnist/',train=True,download=False,\
                                    transform=transforms.ToTensor())
    d_test = torchvision.datasets.FashionMNIST('./data/fmnist/',train=False,download=False,\
                                    transform=transforms.ToTensor())        
else:
    raise ValueError('Unsupported training/testing data')


train = {i: [] for i in range(10)}
for index, (pixels,label) in enumerate(d_train):
    train[label].append(index)
    
test = {i: [] for i in range(10)} 
for index, (pixels,label) in enumerate(d_test):
    test[label].append(index)    

if args.nn_style == 'CNN':
    nchannels = 1
    nclasses = 10
    DANN_F = DANN_CNN_F(nchannels).to(device)
    DANN_C = DANN_CNN_C(nclasses).to(device)
    DANN_D = DANN_CNN_D().to(device)
else:
    raise TypeError('Only CNN at the moment')    

# domain decision
if args.domains == 'binary':
    source_domain = 0
    target_domain = 1
    
    source_train = len(d_train)/2
    target_train = len(d_train)/2    
    
else:
    raise ValueError('No multisource yet')

## how would we do the batch assignments?
# and the batch splitting?

# td = DataLoader(segmentdataset(d_train,range(len(d_train))),batch_size=args.bs,shuffle=True)
td = DataLoader(segmentdataset(d_train,range(1000)),batch_size=args.bs,shuffle=True)






























