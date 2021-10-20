# -*- coding: utf-8 -*-
"""
@author: henry
"""

import torch
from torch import nn,autograd
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
from copy import deepcopy
import torch.nn.functional as F
from utils.mod_optim import FedNova,PSL_FedNova

import gc

class MLP(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP,self).__init__()
        self.layer_input = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)
        
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


class DANN_CNN_F(nn.Module): #invariant features 
    def __init__(self,nchannels):
        super(DANN_CNN_F, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, 32, kernel_size=5)
        self.mpool = nn.MaxPool2d(2,stride=2) 
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mpool(x)
        x = F.relu(self.conv2(x))
        x = self.mpool(x)
        return x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])


class DANN_CNN_C(nn.Module): #classifier
    def __init__(self,nclasses):
        super(DANN_CNN_C,self).__init__()
        self.fc1 = nn.Linear(768,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,nclasses)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

class GradReverse(torch.autograd.Function):#Function):
    def forward(self,x):
        return x.view_as(x)
    
    def backward(self,grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse()(x)

class DANN_CNN_D(nn.Module): #domain
    def __init__(self):
        super(DANN_CNN_D,self).__init__()
        self.fc1 = nn.Linear(768,100)
        self.fc2 = nn.Linear(100,1)

    def forward(self,x):
        x = grad_reverse(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


class vgg11_mod(nn.Module):
    def __init__(self,nchannels,nclasses):
        super(vgg11_mod, self).__init__()
        self.conv1 = nn.Conv2d(nchannels,64,kernel_size=2) #in_channels, out_channels, kernel_size, stride
        # 26x26 
        # maxpool - > 13x13
        self.conv2 = nn.Conv2d(64,128,kernel_size=2)
        # 11x11
        # maxpool -> 5x5 due to floor
        # self.conv3 = nn.Conv2d(128,128,kernel_size=2)
        # 3x3
        # self.conv4 = nn.Conv2d(128,256,kernel_size=2)
        # 1x1
        # self.conv5 = nn.Conv2d(64,64,3)
        # self.conv4 = nn.Conv2d(256,256,3,1)
        # self.conv5 = nn.Conv2d(256,512,3,1)
        # self.conv6 = nn.Conv2d(512,512,3,1)
        # self.conv7 = nn.Conv2d(512,512,3,1)
        # self.conv8 = nn.Conv2d(512,512,3,1)
        # self.conv9 = nn.Conv2d(512,512,3,1)
        # self.dropout1 = nn.Dropout(0.25) #default is 0.5
        # self.dropout2 = nn.Dropout()
        self.fc1 = nn.Linear(73728, 128) #floor([H_in + 2xpad - dilation*(kernel-1)-1]/stride+1)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,nclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2,stride=1)(x) #maxpool2d(kernel_size,stride) ## stride default is kernel_size
        # x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2,stride=1)(x)
        # x = self.dropout2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # input('update')
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        output = F.softmax(x, dim=1)
        return output

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

class Update_DANN_class(object):
    def __init__(self,device,bs,lr,epochs,ldr_train):#,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.epochs = epochs
        self.ldr_train = ldr_train
        self.loss_func = nn.CrossEntropyLoss()
        
        
    def train(self,net_f,net_c):
        net_f.train()
        net_c.train()
        f_optimizer = torch.optim.SGD(net_f.parameters(),lr=self.lr)
        c_optimizer = torch.optim.SGD(net_c.parameters(),lr=self.lr)
        
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            
            # for batch_indx,(images,labels) in enumerate(self.ldr_train):
            images = self.ldr_train[1][0].to(self.device)
            labels = self.ldr_train[1][1].to(self.device)

            net_f.zero_grad()
            net_c.zero_grad()
            
            log_probs = net_c(net_f(images))
            # log_probs = net(images)
            loss = self.loss_func(log_probs,labels)
            loss.backward(retain_graph=True)
            f_optimizer.step()
            c_optimizer.step()
            
            batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net_f,net_f.state_dict(),net_c,net_c.state_dict(),(sum(batch_loss)/len(batch_loss))


def mod_fl_agg(local_w,data_qtys,epochs_vect,lr,global_w):
    tot_data_qty = sum(data_qtys)
    
    device_grads = deepcopy(local_w)
    
    for ind,val in enumerate(local_w):
        for key,val2 in val.items():
            # calculates scaled gradients
            device_grads[ind][key] = (data_qtys[ind]/ (tot_data_qty)) \
                *(global_w[key]-val2)/lr
    
    # vertical sum for device grads
    ovr_grad = deepcopy(local_w[0]) # copy structure
    for key in global_w.keys():
        temp = deepcopy(global_w[key])
        for i in range(len(device_grads)):
            if i == 0:
                temp = device_grads[i][key]
            else:
                temp += device_grads[i][key]
        
        ovr_grad[key] = deepcopy(temp)
    
    # grad desc for new global parameters
    global_w_new = deepcopy(global_w)
    
    for key in global_w.keys():
        global_w_new[key] = global_w[key] - lr*ovr_grad[key]
    
    # print(lr*ovr_grad['fc2.bias'])
    # print(global_w_new['fc2.bias'])
    return global_w_new


def FedAvg2(w,data_qty):
    total_items = int(sum(data_qty))
    try:
        ratios = data_qty/total_items
    except TypeError:
        data_qty = np.array(data_qty)
        ratios = data_qty/total_items
    except:
        data_qty = np.array(data_qty)
        ratios = data_qty/total_items
    
    w_avg = deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(len(w)):
            if i == 0:
                w_avg[k] = w[i][k]*float(ratios[i])
            else:
                w_avg[k] += w[i][k]*float(ratios[i])
            
            # if k == 'features.0.bias':
            #     print(w_avg[k])
            #     input('test')
        #w_avg[k] = torch.div(w_avg[k], len(w))
    
    # print(w_avg['fc2.bias'])
    return w_avg
    
    
    
