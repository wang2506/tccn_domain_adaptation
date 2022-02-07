# -*- coding: utf-8 -*-
"""
@author: henry
"""
import numpy as np
import random
from copy import deepcopy

import torch
from torch import nn,autograd
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP,self).__init__()
        self.layer_input = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)
        #nn.Sigmoid()
        
    def forward(self,x):
        x = x.view(-1,x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNN(nn.Module):
    def __init__(self, nchannels,nclasses):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class segmentdataset(Dataset):
    def __init__(self,dataset,indexes):
        self.dataset = dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self,item):
        image,label = self.dataset[self.indexes[item]]
        return image,label

class LocalUpdate(object):
    def __init__(self,device,bs,lr,epochs,st,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        self.st = st #source/target
        
    def train(self,net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr)#, momentum=0.5,weight_decay=1e-4) #l2 penalty
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                if self.st == 'source':
                    labels = torch.zeros(self.bs,dtype=torch.long).to(self.device)
                elif self.st == 'target':
                    labels = torch.ones(self.bs,dtype=torch.long).to(self.device)
                else:
                    raise TypeError('bad st string')
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))

def wAvg(w):
    #w = [w_at_d1,w_at_d2,etc...]
    w_avg = deepcopy(w[0])
    
    for k in w_avg.keys():
        for i in range(len(w)):
            if i != 0:
                w_avg[k] += w[i][k]

    return w_avg


def test_img(net_g,bs,dset,indx,st,device):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    
    dl = DataLoader(segmentdataset(dset,indx),batch_size=bs,shuffle=True)
    batch_loss = []
    for idx, (data, targets) in enumerate(dl):
        data = data.to(device)
        # target = target.to(device)
        
        if st == 'source':
            targets = torch.zeros(bs,dtype=torch.long).to(device)
        elif st == 'target':
            targets = torch.ones(bs,dtype=torch.long).to(device)
        else:
            raise TypeError('bad st string')        
        
        log_probs = net_g(data)
        batch_loss.append(F.cross_entropy(log_probs, targets, reduction='sum').item())
        
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()

    # test_loss /= len(data_loader.dataset)
    tloss = sum(batch_loss)/len(batch_loss)
    accuracy = 100*correct.item() / len(indx) #data_loader.dataset)
    
    return accuracy, test_loss

    
    
    
