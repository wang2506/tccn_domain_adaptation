# -*- coding: utf-8 -*-
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

class feature_extract(nn.Module):
    def __init__(self):
        super(feature_extract, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        return x
    
class class_classifier(nn.Module):
    def __init__(self):
        super(class_classifier,self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class GradReverse(nn.Module):
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return (-grad_output) #lambda = 1

def grad_reverse(x):
    return GradReverse()(x)

class GRL(nn.Module):
    def __init__(self):
        super(GRL,self).__init__()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,1)

    def forward(self,x):
        x = grad_reverse(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)
# %% 
class LocalUpdate_gr(object):
    def __init__(self,device,bs,lr,epochs,dataset=None,s_inds=None,t_inds=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.dataset = dataset
        self.s_inds = s_inds
        self.t_inds = t_inds
        self.epochs = epochs
        self.s_ldr_train = DataLoader(segmentdataset(self.dataset,self.s_inds),\
                        batch_size=int(self.bs/2),shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        self.d_loss_func = nn.BCELoss()
        
    def train(self,fnet,f2c_net,grl_net):
        fnet.train()
        f2c_net.train()
        grl_net.train()
        
        fnet_optimizer = torch.optim.SGD(fnet.parameters(),lr=self.lr)
        f2c_optimizer = torch.optim.SGD(f2c_net.parameters(),lr=self.lr)
        grl_optimizer = torch.optim.SGD(grl_net.parameters(),lr=self.lr)
        for epoch in range(self.epochs):
            self.t_ldr_train = DataLoader(segmentdataset(self.dataset,self.t_inds),\
                            batch_size=int(self.bs/2),shuffle=True)
            batch_loss = []
            for batch_indx,(s_images,s_labels) in enumerate(self.s_ldr_train):
                if len(s_images) != int(self.bs/2):
                    break
                t_images,_ = next(iter(self.t_ldr_train))
                if len(t_images) != int(self.bs/2):
                    break
                s_images,s_labels = s_images.to(self.device),s_labels.to(self.device)
                t_images = t_images.to(self.device)
                
                # 1 - train feature extractor and class classifier on source batch
                fnet.zero_grad()
                f2c_net.zero_grad()
                class_outs = f2c_net(fnet(s_images))
                loss1 = self.loss_func(class_outs,s_labels)
                loss1.backward(retain_graph=True)
                f2c_optimizer.step()                
                fnet_optimizer.step()

                # 2 - train feature extractor and domain classifier on full batch 
                fnet.zero_grad()                
                grl_net.zero_grad()
                sum_images = torch.cat([s_images, t_images], 0)
                sum_labels = torch.cat([torch.zeros(int(self.bs/2)),\
                                torch.ones(int(self.bs/2))],0)
                domain_outs = grl_net(fnet(sum_images))
                loss2 = self.d_loss_func(domain_outs,sum_labels[:,None].to(self.device))
                loss2.backward(retain_graph=True)
                grl_optimizer.step()
                fnet_optimizer.step()
                
        return fnet,fnet.state_dict(), f2c_net,f2c_net.state_dict(), \
            grl_net,grl_net.state_dict()

def train_gr(s_dset,t_dset,args,d_train,nnet1,nnet2,nnet3,device):
    # print('starting the training for new device with labeled data')
    train_obj = LocalUpdate_gr(device=torch.device(device),\
                bs=args.div_bs,lr=args.div_lr, \
                epochs=args.st_time,dataset=d_train,s_inds=s_dset,t_inds=t_dset)
    
    _,fnet_w,_,f2c_w,_,grl_w = train_obj.train(nnet1,nnet2,nnet3)
    return fnet_w,f2c_w,grl_w

def test_img_gr(net1,net2,bs,dset,indx,device):
    net1.eval()
    net2.eval()
    # testing
    test_loss = 0
    correct = 0
    
    dl = DataLoader(segmentdataset(dset,indx),batch_size=bs,shuffle=True)
    batch_loss = []
    for idx, (data, targets) in enumerate(dl):
        data = data.to(device)
        targets = targets.to(device)
        log_probs = net2(net1(data))
        batch_loss.append(F.cross_entropy(log_probs, targets, reduction='sum').item())
        
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()

    tloss = sum(batch_loss)/len(batch_loss)
    accuracy = 100*correct.item() / len(indx)
    
    return accuracy, test_loss


# %%
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
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    
# %%
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
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr)
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

    tloss = sum(batch_loss)/len(batch_loss)
    accuracy = 100*correct.item() / len(indx)
    
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
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr)
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

    tloss = sum(batch_loss)/len(batch_loss)
    accuracy = 100*correct.item() / len(indx)
    
    return accuracy, test_loss
