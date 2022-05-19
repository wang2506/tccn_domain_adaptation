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


class GCNN(nn.Module):
    def __init__(self, nchannels,nclasses):
        super(GCNN, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64,128,kernel_size=5)
        self.drop = nn.Dropout2d()
        self.mpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048,nclasses)

    def forward(self, x):
        x = F.relu(self.mpool(self.drop(self.conv1(x))))
        x = F.relu(self.mpool(self.drop(self.conv2(x))))
        x = F.relu(self.drop(self.conv3(x)))
        # print(x.shape[1]*x.shape[2]*x.shape[3])
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.drop(x)#, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.drop(x)#, training=self.training)
        x = self.fc3(x)
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
                    labels = torch.zeros(len(labels),dtype=torch.long).to(self.device)
                elif self.st == 'target':
                    labels = torch.ones(len(labels),dtype=torch.long).to(self.device)
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
            if i == 0:
                w_avg[k] = w[i][k]/len(w)
            else:# i != 0:
                w_avg[k] += w[i][k]/len(w)
    return w_avg

def wAvg_weighted(w,weights):
    w_avg = deepcopy(w[0])
    
    for k in w_avg.keys():
        for i in range(len(w)):
            if i == 0:
                w_avg[k] = w[i][k]*weights[i]
            else:# i != 0:
                w_avg[k] += w[i][k]*weights[i]
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

# %% fxn to evaluate hypothesis mismatch
def d2d_mismatch_test(c_net_s,params_t,params_s,dset,indx_t,bs,device):
    c_net_t = deepcopy(c_net_s)
    c_net_s.load_state_dict(params_s)
    c_net_t.load_state_dict(params_t)
    c_net_t.eval()
    c_net_s.eval()
    
    matches = 0
    dl = DataLoader(segmentdataset(dset,indx_t),batch_size=bs,shuffle=True)
    for idx, (data,targets) in enumerate(dl):
        data = data.to(device)
        
        log_probs_s = c_net_s(data)
        log_probs_t = c_net_t(data)

        s_pred = log_probs_s.data.max(1,keepdim=True)[1]
        t_pred = log_probs_t.data.max(1,keepdim=True)[1]
        
        matches += s_pred.eq(t_pred).long().cpu().sum()
    acc = 100*matches.item() / len(indx_t)
    diff = (100-acc)/100
    return diff

# %% neural nets for the optim problem + source error calc
class LocalUpdate_strain(object):
    def __init__(self,device,bs,lr,epochs,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self,net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr)#, momentum=0.5,weight_decay=1e-4) #l2 penalty
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))

def init_source_train(ld_set,args,d_train,nnet,device):
    # print('starting the training for new device with labeled data')
    train_obj = LocalUpdate_strain(device=torch.device(device),\
                bs=args.div_bs,lr=args.div_lr, \
                epochs=args.st_time,dataset=d_train,indexes=ld_set)
    
    _,c_w,loss = train_obj.train(nnet)
    
    return c_w,loss
    


def test_img_strain(net_g,bs,dset,indx,device):
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
