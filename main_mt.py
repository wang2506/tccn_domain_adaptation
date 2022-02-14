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


from optim_prob.optim_utils.optim_parser import optim_parser

cwd = os.getcwd()
oargs = optim_parser()


# %% load in optimization and divergence results
with open(cwd+'/optim_results/psi_val/devices'+str(oargs.t_devices)+\
          '_seed'+str(oargs.seed)+'_'+oargs.dset_type\
            +'_'+oargs.labels_type,'rb') as f:
    psi_vals = pk.load(f)

with open(cwd+'/optim_results/alpha_val/devices'+str(oargs.t_devices)+\
          '_seed'+str(oargs.seed)+'_'+oargs.dset_type\
            +'_'+oargs.labels_type,'rb') as f:
    alpha_vals = pk.load(f)

## can and should obtain the actual hat_ep_val for each of the devices with data




# %% load in datasets




# %% train sources



# %% transfer to targets + record results






