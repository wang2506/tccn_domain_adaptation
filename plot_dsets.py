# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:22:19 2023

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
from optim_prob.mnist_m import MNISTM

import os

cwd = os.getcwd()

mnist = torchvision.datasets.MNIST(cwd+'/data/',train=True,download=True,\
                transform=transforms.ToTensor())

tx_dat = torchvision.transforms.Compose([transforms.ToTensor()])#,transforms.Pad(14-8)])
try: 
    usps = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                    transform=tx_dat)
except:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context            
    usps = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                    transform=tx_dat)    


tx_dat =  torchvision.transforms.Compose([transforms.ToTensor()])
mnistm = MNISTM(cwd+'/data/',train=True,download=True,\
                 transform=tx_dat)

# %%
m_targets = mnist.targets
u_targets = np.array(usps.targets)
mm_targets = mnistm.targets

## select one of each
m_imgs = []
u_imgs = []
mm_imgs = []
# for i in range(4):
for i in [0,3,5,8,9]:
    m_imgs.append(mnist[np.where(m_targets == i)[0][10]][0])
    u_imgs.append(usps[np.where(u_targets == i)[0][10]][0])
    mm_imgs.append(mnistm[np.where(mm_targets == i)[0][10]][0])




# %% 
# import cv2
import matplotlib.pyplot as plt
fig,ax = plt.subplots(3,5,figsize=(5,3.5),dpi=500)#,bbox_inches='tight')
fig.tight_layout(pad=-1.5)
for i,j in enumerate(range(0,5,1)):
    ax[0,i].imshow(m_imgs[j][0],cmap='gray')
    ax[0,i].set_xticks([])
    ax[0,i].set_yticks([])    
    
    ax[1,i].imshow(u_imgs[j][0],cmap='gray')
    ax[1,i].set_xticks([])
    ax[1,i].set_yticks([])

    # # temp = np.dstack((c_imgs[j][0],c_imgs[j][1],c_imgs[j][2]))
    ax[2,i].imshow(mm_imgs[j].permute(1, 2, 0))#temp) #[c_imgs[j][0],c_imgs[j][1],c_imgs[j][2]])
    ax[2,i].set_xticks([])
    ax[2,i].set_yticks([])

for i,j in enumerate([0,3,5,8,9]):
    ax[2,i].set_xlabel('Label:'+str(j))


ax[0,0].set_ylabel('MNIST')
ax[1,0].set_ylabel('USPS')
ax[2,0].set_ylabel('MNIST-M')
    
# %%
plt.savefig(cwd+'/data/dset_pic.pdf',dpi=1000,bbox_inches='tight')
plt.savefig(cwd+'/data/dset_pic.png',dpi=1000,bbox_inches='tight')



    