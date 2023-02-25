# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:18:01 2022

@author: ch5b2
"""
import pickle as pk
import os
import numpy as np
import random

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

# from comp_iclr_utils import Generator, Disentangler, Classifier, \
#     Feature_Discriminator, Reconstructor, Mine
from utils.iclr_utils import Generator, Disentangler, Classifier, \
    Feature_Discriminator, Reconstructor, Mine
from optim_prob.optim_utils.optim_parser import optim_parser
from optim_prob.mnist_m import MNISTM
from sklearn.cluster import KMeans


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

# %%
class iclr_method(object):
    def __init__(self,args):
        self.args = args
        device = torch.device('cuda:'+str(args.div_gpu_num))
        self.device = device
        self.coefficient_matrix=[[np.round(1/args.l_devices,2)]*args.l_devices \
            for i in range(args.u_devices)]
        self.pre_inertia = -1
        self.inertia = [0]*args.l_devices
        self.src_domain_code = np.repeat(np.array([[*([1]), *([0])]]), args.div_bs, axis=0)
        self.tgt_domain_code = np.repeat(np.array([[*([0]), *([1])]]), args.div_bs, axis=0)
        self.src_domain_code = Variable(\
            torch.FloatTensor(self.src_domain_code).to(device), requires_grad=False)
        self.tgt_domain_code = Variable(\
            torch.FloatTensor(self.tgt_domain_code).to(device), requires_grad=False)
        self.softmax = lambda z:np.exp(z)/np.sum(np.exp(z))
        print('pulling datasets and quantities')   
        self.dataset_s = []
        self.dataset_test_s = []
        cwd = os.getcwd()
        
        if args.grad_rev == True:
            end2 = 'gr'
        else:
            end2 = ''
        
        if args.fl == True:
            prefl = 'fl'
        else:
            prefl = ''
        
        
        if args.dset_split == 0:
            if args.dset_type == 'MM':
                end = '_base_6'
            else:
                end = ''
                
            with open(cwd+'/optim_prob/data_div/devices'+str(args.t_devices)\
                +'_seed'+str(args.seed)+'_'+args.div_nn\
                +'_'+args.dset_type+'_'+args.labels_type+'_lpd','rb') as f:
                lpd = pk.load(f)
            with open(cwd+'/optim_prob/data_div/devices'+str(args.t_devices)\
                +'_seed'+str(args.seed)+'_'+args.div_nn\
                +'_'+args.dset_type+'_'+args.labels_type+'_dindexsets','rb') as f:
                d_dsets = pk.load(f) 
        
        elif args.dset_split == 1:
            pre = ''
            
            with open(cwd+'/optim_prob/data_div/devices'+str(args.t_devices)\
                +'_seed'+str(args.seed)+'_'+args.div_nn\
                +'_'+args.dset_type+'_'+args.labels_type+'_lpd','rb') as f:
                lpd = pk.load(f)
            with open(cwd+'/optim_prob/data_div/devices'+str(args.t_devices)\
                +'_seed'+str(args.seed)+'_'+args.div_nn\
                +'_'+args.dset_type+'_'+args.labels_type+'_dindexsets','rb') as f:
                d_dsets = pk.load(f)             
            
            raise TypeError('tbd')
        elif args.dset_split == 2:
            pre = 'total_'
            raise TypeError('tbd')
            # with open(cwd+'/optim_prob/data_div/'+pre+'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            #     +'_'+args.div_nn\
            #     +'_'+args.split_type+'_'+args.labels_type+'_lpd','rb') as f:
            #     lpd = pk.load(f)
            # with open(cwd+'/optim_prob/data_div/'+pre+'devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            #     +'_'+args.div_nn\
            #     +'_'+args.split_type+'_'+args.labels_type+'_dindexsets','rb') as f:
            #     d_dsets = pk.load(f)
        
        self.d_dsets = d_dsets #image indexes
        
        
        print('dataset + minibatch formatting')
        if args.dset_split == 0:
            if args.dset_type == 'M':
                print('Using MNIST \n')
                d_train = torchvision.datasets.MNIST(cwd+'/data/',train=True,download=True,\
                                transform=transforms.ToTensor())
            elif args.dset_type == 'S': #needs scipy
                print('Using SVHN \n')
                tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),\
                                transforms.Grayscale(),transforms.CenterCrop(28)])
                d_train = torchvision.datasets.SVHN(cwd+'/data/svhn',split='train',download=True,\
                                transform=tx_dat)
                d_train.targets = d_train.labels
                #http://ufldl.stanford.edu/housenumbers/
            elif args.dset_type == 'U':
                print('Using USPS \n')
                tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])
                try: 
                    d_train = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                                    transform=tx_dat)
                except:
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context            
                    d_train = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                                    transform=tx_dat)    
            elif args.dset_type == 'MM':
                print('Using MNIST-M \n')
                tx_dat =  torchvision.transforms.Compose([transforms.ToTensor()])
                d_train = MNISTM(cwd+'/data/',train=True,download=True,\
                                 transform=tx_dat)                    
            else:
                raise TypeError('Dataset exceeds sims')
        else: #if oargs.dset_split == 1: 
            tx_m = torchvision.transforms.Compose([transforms.ToTensor()])
            tx_mm = torchvision.transforms.Compose([transforms.ToTensor(),\
                        transforms.Grayscale()])
            tx_u = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])    
            
            d_m = torchvision.datasets.MNIST(cwd+'/data/',train=True,download=True,\
                            transform=tx_m)
            d_mm = MNISTM(cwd+'/data/',train=True,download=True,\
                             transform=tx_mm)        
            try: 
                d_u = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                                transform=tx_u)
            except:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context            
                d_u = torchvision.datasets.USPS(cwd+'/data/',train=True,download=True,\
                                transform=tx_u)
            d_u.targets = torch.tensor(d_u.targets)    
            
            if args.dset_split == 1:
                if args.split_type == 'M+MM':
                    print('Using MNIST + MNIST-M')
                    d_train = d_m+d_mm
                    d_train.targets = torch.concat([d_m.targets,d_mm.targets])
                elif args.split_type == 'M+U':
                    print('Using MNIST + USPS')
                    d_train = d_m+d_u
                    d_train.targets = torch.concat([d_m.targets,d_u.targets])
                elif args.split_type == 'MM+U':
                    print('Using MNIST-M + USPS')
                    d_train = d_mm+d_u       
                    d_train.targets = torch.concat([d_mm.targets,d_u.targets])
                elif args.split_type == 'A':
                    print('Using MNIST + MNIST-M + USPS')
                    d_train = d_m+d_mm+d_u
                    d_train.targets = torch.concat([d_m.targets,d_mm.targets,d_u.targets])
                else:
                    raise TypeError('Datasets exceed sims')
            elif args.dset_split == 2: 
                with open(cwd+'/optim_prob/data_div/d2dset_devices'+str(args.t_devices)+\
                          '_seed'+str(args.seed),'rb') as f:
                    d2dset = pk.load(f)
                d_train_dict = {}
                
                if args.split_type == 'M+MM':
                    d0,d1= d_m,d_mm
                elif args.split_type == 'M+U':
                    d0,d1 = d_m,d_u
                elif args.split_type == 'MM+U':
                    d0,d1 = d_mm,d_u
                else:
                    raise TypeError('datasets exceed sims')
                
                for ind,dc in enumerate(np.where(d2dset==0)[0]):
                    d_train_dict[dc] = d0
                for ind,dc in enumerate(np.where(d2dset==1)[0]):
                    d_train_dict[dc] = d1
                d_train_dict = dict(sorted(d_train_dict.items()))
        
                print('faulty stuff within this loop')
        
        data_qty_alld,split_lqtys,split_uqtys = 0,0,0
        td_dict = {'_data_qty':data_qty_alld,'_split_lqtys':split_lqtys,\
                   '_split_uqtys':split_uqtys}
        for ie,entry in enumerate(td_dict.keys()):
            with open(cwd+'/optim_prob/data_div/devices'+str(args.t_devices)\
                +'_seed'+str(args.seed)+entry\
                +'_'+args.avg_size,'rb') as f:
                td_dict[entry] = pk.load(f)
        data_qty_alld = td_dict['_data_qty']
        split_lqtys = td_dict['_split_lqtys']
        split_uqtys = td_dict['_split_uqtys']
        
        ld_sets = {}
        for i in d_dsets.keys():
            if i >= args.l_devices:
                break
            else:
                ld_sets[i] = random.sample(self.d_dsets[i],split_lqtys[i])
        self.ld_sets = ld_sets
        
        ## this assumes that you have already determined the source/target split
        ## HARD CODED!!!
        ## TODO 
        device_datasets_batched = {}
        for i in range(args.l_devices):
            if args.dset_split > 1: 
                device_datasets_batched[i] = DataLoader(segmentdataset(d_train_dict[i],ld_sets[i]),\
                    batch_size=args.div_bs,shuffle=True)
            else:
                device_datasets_batched[i] = DataLoader(segmentdataset(d_train,ld_sets[i]),\
                    batch_size=args.div_bs,shuffle=True)
        self.dd_batched = device_datasets_batched
        
        ud_batched = {}
        for i in range(args.u_devices):
            if args.dset_split > 1:
                ud_batched[i] = DataLoader(segmentdataset(d_train_dict[args.l_devices+i],\
                    d_dsets[args.l_devices+i]),\
                    batch_size=args.div_bs,shuffle=True)
            else:
                ud_batched[i] = DataLoader(segmentdataset(d_train,\
                    d_dsets[args.l_devices+i]),\
                    batch_size=args.div_bs,shuffle=True)
        self.ud_batched = ud_batched
        print('dataset + batch formatting finished')
        
        print('building models')
        self.G_s = []
        self.C_s = []
        self.FD = []
        self.D = []
        self.DC = []
        self.R = []
        self.M = []

        for i, j in enumerate(range(args.l_devices)):
            self.G_s.append(Generator(nchannels=1)) ## source generators, (feature extractors) G_i
            self.C_s.append(Classifier()) ## source classifiers, C_i 
            self.FD.append(Feature_Discriminator()) ## domain identifiers, DI_i
            self.D.append(Disentangler()) ## source disentanglers, (separates the domain invariant and domain specific features), D_i
            self.DC.append(Classifier()) ## K-way (class) classifier CI_i
            self.R.append(Reconstructor()) ## recombine output of the disentangler - unused for targets
            self.M.append(Mine()) ## MINE estimator, M_i
        
        self.G_t = []
        self.C_t = []
        for i in range(args.u_devices):
            self.G_t.append( Generator(nchannels=1)) ## target generator
            self.C_t.append( Classifier()) ## target classifier
        print('building models finished')
        
        for G_s, C_s, FD, D, DC, R, M in zip(self.G_s, self.C_s, self.FD, \
                            self.D, self.DC, self.R, self.M):
            G_s.to(device)
            C_s.to(device)
            FD.to(device)
            D.to(device)
            DC.to(device)
            R.to(device)
            M.to(device)
        
        for G_t, C_t in zip(self.G_t,self.C_t):
            G_t.to(device)
            C_t.to(device)
        
        # setting optimizer
        self.opt_g_s = []
        self.opt_c_s = []
        self.opt_fd = []
        self.opt_d = []
        self.opt_dc = []
        self.opt_r = []
        self.opt_m = []
        
        for G_s, C_s, FD, D, DC, R, M in zip(self.G_s, self.C_s, self.FD, self.D, \
                self.DC, self.R, self.M):
            self.opt_g_s.append(optim.SGD(G_s.parameters(), lr=args.div_lr))
            self.opt_c_s.append(optim.SGD(C_s.parameters(), lr=args.div_lr))
            self.opt_fd.append(optim.SGD(FD.parameters(), lr=args.div_lr))
            self.opt_d.append(optim.SGD(D.parameters(), lr=args.div_lr))
            self.opt_dc.append(optim.SGD(DC.parameters(), lr=args.div_lr))
            self.opt_r.append(optim.SGD(R.parameters(), lr=args.div_lr))
            self.opt_m.append(optim.SGD(M.parameters(), lr=args.div_lr))
        
        self.opt_g_t = []
        self.opt_c_t = []
        for G_t,C_t in zip(self.G_t,self.C_t):
            self.opt_g_t.append(optim.SGD(G_t.parameters(), lr=args.div_lr))
            self.opt_c_t.append(optim.SGD(C_t.parameters(), lr=args.div_lr))
        
        print('check')
        # initilize parameters
        for G in self.G_s:
            for G_t in self.G_t:
                for net, net_cardinal in zip(G.named_parameters(), G_t.named_parameters()):
                    net[1].data = net_cardinal[1].data.clone()
        for C in self.C_s:
            for C_t in self.C_t:
                for net, net_cardinal in zip(C.named_parameters(), C_t.named_parameters()):
                    net[1].data = net_cardinal[1].data.clone()
    
    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def mutual_information_estimator(self, index, x, y, y_):
        joint,marginal = self.M[index](x,y), self.M[index](x,y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def reconstruct_loss(self, src, tgt):
        return torch.sum((src-tgt)**2) / (src.shape[0] * src.shape[1])

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))    
    
    def group_step(self, step_list):
        for i in range(len(step_list)):
            step_list[i].step()
        self.reset_grad()
    
    def reset_grad(self):
        for (opt_g_s,opt_c_s,opt_fd, opt_d, opt_dc, opt_m,opt_r) in \
                zip(self.opt_g_s, self.opt_c_s, self.opt_fd, self.opt_d, self.opt_dc, self.opt_m,self.opt_r):
            opt_g_s.zero_grad()
            opt_c_s.zero_grad()
            opt_fd.zero_grad()
            opt_d.zero_grad()
            opt_dc.zero_grad()
            opt_m.zero_grad()
            opt_r.zero_grad()
        
        for (opt_c_t,opt_g_t) in \
            zip(self.opt_c_t,self.opt_g_t):
            opt_c_t.zero_grad()
            opt_g_t.zero_grad()

    def train(self,epoch):
        criterion=nn.CrossEntropyLoss().to(device)
        adv_loss = nn.BCEWithLogitsLoss().to(device)

        for G, C, FD, D, DC, R, M in \
            zip(self.G_s, self.C_s, self.FD, self.D, self.DC, self.R, self.M):
            # for i in range(len(G)):
            G.train()
            C.train()
            FD.train()
            D.train()
            DC.train()
            R.train()
            M.train()
            
        for Gt, Ct in zip(self.G_t, self.C_t):
            Gt.train()
            Ct.train()
        
        for t_d in range(args.u_devices):
            zip_struct = []
            for i in range(args.l_devices):
                zip_struct.append(list(self.dd_batched[i]))
            zip_struct.append(list(self.ud_batched[t_d]))
            
            mb_sizes = []
            for i in zip_struct:
                mb_sizes.append(len(i))
            t_batches = min(mb_sizes)-1
            
            loss_record = []
            loss_t_record = []
            
            for c_batch in range(t_batches):
                for i in range(args.l_devices):
                    data_s,label_s = zip_struct[i][c_batch][0].to(device),zip_struct[i][c_batch][-1].to(device)
                    data_t,label_t = zip_struct[-1][c_batch][0].to(device),zip_struct[-1][c_batch][-1].to(device)
                    self.reset_grad()
                    feat = self.G_s[i](data_s)
                    feat_fc2 = feat['f_fc2']
                    output = self.C_s[i](feat_fc2)

                    feat_t = self.G_t[t_d](data_t)
                    feat_fc2_t = feat_t['f_fc2']
                    output_t = self.C_t[t_d](feat_fc2_t)

                    loss_s = criterion(output, label_s)
                    loss_t = criterion(output_t, label_t)
                    loss_record.append(loss_s.item())
                    loss_t_record.append(loss_t.item())
    
                    loss_s.backward(retain_graph=True)

                    feat_conv3 = feat['f_conv3']
                    output_d=self.DC[i](self.D[i](feat_conv3))
                    loss_s = criterion(output, label_s)
                    loss_dis = self.discrepancy(output,output_d)
                    loss_s -= loss_dis
                    loss_s.backward(retain_graph=True)
                    
                    for net1, net2 in \
                        zip(self.G_s[i].named_parameters(), self.G_t[t_d].named_parameters()):
                        if net1[1].grad is not None:
                            net2[1].grad = net1[1].grad.clone() * self.coefficient_matrix[t_d][i] 
                    for net1, net2 in \
                        zip(self.C_s[i].named_parameters(), self.C_t[t_d].named_parameters()):
                        if net1[1].grad is not None:
                            net2[1].grad = net1[1].grad.clone() * self.coefficient_matrix[t_d][i] 
                    self.group_step([self.opt_d[i], self.opt_dc[i], self.opt_g_t[t_d], \
                            self.opt_c_t[t_d], self.opt_g_s[i], self.opt_c_s[i]])
                    
                    features_list = self.G_s[i](data_s)
                    dis_features = self.D[i](features_list['f_conv3'])
                    dis_features_shuffle = torch.index_select(dis_features,0,\
                        Variable(torch.randperm(dis_features.shape[0]).to(device)))
                    features_fc2 = features_list['f_fc2']
                    MI = self.mutual_information_estimator(i, features_fc2, \
                        dis_features, dis_features_shuffle) / args.div_bs
                    entropy_s = self.ent(dis_features) / args.div_bs
                    recon_features = self.R[i](torch.cat((features_fc2, dis_features),1)) \
                        / args.div_bs
                    recon_loss =  self.reconstruct_loss(features_list['f_conv3'], \
                        recon_features) / args.div_bs
                    total_loss = (MI - entropy_s + recon_loss ) * 0.1
                    total_loss.backward()
                    self.group_step([self.opt_d[i], self.opt_r[i], \
                                     self.opt_m[i], self.opt_g_s[i]])

                    test_batch = zip_struct[-1][np.random.randint(0,len(zip_struct[-1])-1)]
                    test_img = test_batch[0].to(device)
                    # img = Variable(img.to(device))
                    feat_torch = self.G_t[t_d](test_img)['f_fc2']
                    feat = feat_torch.data.cpu().numpy()

                    kmeans = KMeans(n_clusters=10, max_iter=1000).fit(feat)

                    cur_inertia = kmeans.inertia_ / 1e3
                    if self.pre_inertia == -1:
                        self.pre_inertia = cur_inertia
                        continue
                    else:
                        inertia_gain = (self.pre_inertia - cur_inertia)\
                            /self.coefficient_matrix[t_d][i]*(np.round(1/args.l_devices,2))
                        self.inertia[i] = inertia_gain + self.inertia[i]
                        self.pre_inertia = cur_inertia

                    src_domain_pred = self.FD[i](self.G_s[i](data_s)['f_fc2'])
                    tgt_domain_pred = self.FD[i](self.G_t[t_d](data_t)['f_fc2'])

                    df_loss_src = adv_loss(src_domain_pred, self.src_domain_code)
                    df_loss_tgt = adv_loss(tgt_domain_pred, self.tgt_domain_code)

                    loss = (df_loss_src + df_loss_tgt)/args.div_bs
                    loss.backward()
                    self.group_step([self.opt_fd[i], self.opt_g_s[i], self.opt_g_t[t_d]])

                    src_domain_pred = self.FD[i](self.G_s[i](data_s)['f_fc2'])
                    tgt_domain_pred = self.FD[i](self.G_t[t_d](data_t)['f_fc2'])
    
                    df_loss_src = adv_loss(src_domain_pred, 1-self.src_domain_code)
                    df_loss_tgt = adv_loss(tgt_domain_pred, 1-self.tgt_domain_code)
                    loss = (df_loss_src + df_loss_tgt)/args.div_bs
                    loss.backward()
                    self.group_step([self.opt_fd[i], self.opt_g_s[i], self.opt_g_t[t_d]])

            coefficient_matrix_diff = self.softmax(self.inertia)
            tc_mat_diff = [0.20 + 0.2 * tmp for tmp in coefficient_matrix_diff]
            self.coefficient_matrix[t_d] = [tcmd/sum(tc_mat_diff) \
                for tcmd in tc_mat_diff]
        print(self.coefficient_matrix)
        
        return self.coefficient_matrix

# %%
if __name__ == '__main__':
    args = optim_parser()
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    test = iclr_method(args)

    results = test.train(100)
    
    cwd = os.getcwd()
    import pickle as pk
    with open(cwd+'/base_results/init_test_complexnets','wb') as f:
        pk.dump(results,f)


