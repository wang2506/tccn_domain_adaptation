# -*- coding: utf-8 -*-
import numpy as np
import random
from copy import deepcopy

import torch
from torch import nn,autograd
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

def rescale_alphas(c_psi,c_alpha):
    # sources received models (alpha) adjust
    s_pv = np.where(np.array(c_psi) == 0)[0]
    s_alpha = c_alpha[:,s_pv]
    s_alpha[np.where(s_alpha <= 5e-2)] = 0 #2.5
    
    s_alpha_sums = np.sum(s_alpha,axis=0)
    for div_factor in s_alpha_sums:
        if div_factor > 0:
            raise ValueError('Should not have greater than 0')
    
    # targets received models (alpha) adjust
    t_pv = np.where(np.array(c_psi) == 1)[0]
    t_alpha = c_alpha[:,t_pv]
    t_alpha[np.where(t_alpha <= 1e-2)] = 0 
    
    t_alpha_sums = np.sum(t_alpha,axis=0)
    for ind_f,div_factor in enumerate(t_alpha_sums):
        if div_factor > 0:
            # print(div_factor)
            t_alpha[:,ind_f] = [np.round(c_val/div_factor,2) for c_val in t_alpha[:,ind_f]]

    # concatenate the two alpha matrices based on numerical order
    for a_ind in range(c_alpha.shape[0]):
        if a_ind == 0:
            if a_ind in s_pv:
                ovr_alpha = s_alpha[:,np.where(s_pv==a_ind)[0]]
            else:
                ovr_alpha = t_alpha[:,np.where(t_pv==a_ind)[0]]
        else:
            if a_ind in s_pv:
                ovr_alpha = np.hstack((ovr_alpha,s_alpha[:,np.where(s_pv==a_ind)[0]]))
            else:
                ovr_alpha = np.hstack((ovr_alpha,t_alpha[:,np.where(t_pv==a_ind)[0]]))
        
    return s_alpha,t_alpha,ovr_alpha,s_pv,t_pv


class segmentdataset(Dataset):
    def __init__(self,dataset,indexes):
        self.dataset = dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self,item):
        image,label = self.dataset[self.indexes[item]]
        return image,label

def test_img_ttest(net_g,bs,dset,indx,device):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    
    dl = DataLoader(segmentdataset(dset,indx),batch_size=bs,shuffle=True)
    batch_loss = []
    for idx, (data, targets) in enumerate(dl):
        data = data.to(device)
        targets = targets.to(device)
        log_probs = net_g(data)
        batch_loss.append(F.cross_entropy(log_probs, targets, reduction='sum').item())
        
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()

    # test_loss /= len(data_loader.dataset)
    tloss = sum(batch_loss)/len(batch_loss)
    accuracy = 100*correct.item() / len(indx) #data_loader.dataset)
    
    return accuracy, test_loss

def alpha_avg(w,alphas):
    #w = {w_at_d1,w_at_d2,etc...}
    sources = w.keys()
    start = 1
    for i,j in enumerate(sources):
        if start == 1:
            w_avg = deepcopy(w[j])
            start = 0
        for k in w_avg.keys():
            if i == 0:
                w_avg[k] = w[j][k]*alphas[j]
            else:
                w_avg[k] += w[j][k]*alphas[j]
    return w_avg
