# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:47:48 2021

@author: henry
"""
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms


def __getDirData__(data, psizes, seed, alpha):
    n_nets = len(psizes)
    K = 10
    labelList = np.array(data.train_labels) ##data.targets
    min_size = 0
    N = len(labelList)
    np.random.seed(2020)

    net_dataidx_map = {}
    while min_size < K:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(labelList == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print('Data statistics: %s' % str(net_cls_counts))

    local_sizes = []
    for i in range(n_nets):
        local_sizes.append(len(net_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes/np.sum(local_sizes)
    print(weights)

    return idx_batch, weights

if __name__ == '__main__':
    d_train = torchvision.datasets.FashionMNIST('../data/fmnist/',train=True,download=False,\
                                transform=transforms.ToTensor())
    d_test = torchvision.datasets.FashionMNIST('../data/fmnist/',train=False,download=False,\
                                    transform=transforms.ToTensor())
    
    seed = 1
    alpha = 0.05 #controls non-iidness of the dataset
    size = 20 #args.devices
    
    # this is same dataset size for all devices for now
    partition_sizes = [1.0 / size for _ in range(size)] # this should also be non-iid
    a,b = __getDirData__(data=d_train, psizes=partition_sizes, seed=seed, alpha=alpha)