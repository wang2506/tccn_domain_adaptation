# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle as pk
import random
from copy import deepcopy
import scipy 
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

from optim_utils.optim_parser import optim_parser
from div_utils.neural_nets import MLP,CNN,segmentdataset,\
    LocalUpdate, wAvg, test_img, GCNN
from mnist_m import MNISTM

cwd = os.getcwd()
args = optim_parser()
seed = args.seed
np.random.seed(seed)
random.seed(seed)

if args.label_split == 0:
    args.labels_type = 'iid'

# %% imports + redefinitions
# comp method
if args.div_comp == 'gpu':
    device = torch.device('cuda:'+str(args.div_gpu_num))
elif args.div_comp == 'cpu':
    device = torch.device('cpu')
else:
    raise TypeError('wrong computation method')

# data qty
# try: 
#     with open(cwd+'/data_div/devices'+str(args.t_devices)\
#               +'_seed'+str(args.seed)+'_data_qty_'\
#             +args.avg_size,'rb') as f:
#         alld_qty = pk.load(f)
# except:
all_u_qtys = np.random.normal(args.avg_uqty,args.avg_uqty/6,\
            size=args.u_devices).astype(int) #all_unlabelled_qtys
split_lqtys = np.random.normal(args.avg_lqty_l,args.avg_lqty_l/6,\
            size=args.l_devices).astype(int)
split_uqtys = np.random.normal(args.avg_lqty_u,args.avg_lqty_u/6,\
            size=args.l_devices).astype(int)
net_l_qtys = split_lqtys + split_uqtys
data_qty_alld = list(net_l_qtys)+list(all_u_qtys)    

alld_qty = data_qty_alld

td_dict = {'_data_qty':data_qty_alld,'_split_lqtys':split_lqtys,\
           '_split_uqtys':split_uqtys}        
for ie,entry in enumerate(td_dict.keys()):
    with open(cwd+'/data_div/devices'+str(args.t_devices)+\
              '_seed'+str(args.seed)+entry\
            +'_'+args.avg_size,'wb') as f:
        pk.dump(td_dict[entry],f)

# dataset determination
pwd = os.path.dirname(cwd)
if args.dset_split == 0:
    if args.dset_type == 'M':
        print('Using MNIST \n')
        d_train = torchvision.datasets.MNIST(pwd+'/data/',train=True,download=True,\
                        transform=transforms.ToTensor())
    elif args.dset_type == 'S': #needs scipy
        print('Using SVHN \n')
        tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),\
                        transforms.Grayscale(),transforms.CenterCrop(28)])
        d_train = torchvision.datasets.SVHN(pwd+'/data/svhn/',split='train',download=True,\
                        transform=tx_dat)
        d_train.targets = d_train.labels
        #http://ufldl.stanford.edu/housenumbers/
    elif args.dset_type == 'U':
        print('Using USPS \n')
        tx_dat = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])
        try: 
            d_train = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                            transform=tx_dat)
        except:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context            
            d_train = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                            transform=tx_dat)
        d_train.targets = np.array(d_train.targets)
    elif args.dset_type == 'MM':
        print('Using MNIST-M \n')
        tx_dat =  torchvision.transforms.Compose([transforms.ToTensor()])#,\
                    # transforms.Grayscale()])
        d_train = MNISTM(pwd+'/data/',train=True,download=True,\
                         transform=tx_dat)
    else:
        raise TypeError('Dataset exceeds sims')
else: 
    tx_m = torchvision.transforms.Compose([transforms.ToTensor()])
    tx_mm = torchvision.transforms.Compose([transforms.ToTensor(),\
                transforms.Grayscale()])
    tx_u = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Pad(14-8)])    
    
    d_m = torchvision.datasets.MNIST(pwd+'/data/',train=True,download=True,\
                    transform=tx_m)
    d_mm = MNISTM(pwd+'/data/',train=True,download=True,\
                     transform=tx_mm)        
    try: 
        d_u = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
                        transform=tx_u)
    except:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context            
        d_u = torchvision.datasets.USPS(pwd+'/data/',train=True,download=True,\
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
        d2dset = np.random.randint(0,2,size=args.t_devices)
        with open(cwd+'/data_div/d2dset_devices'+str(args.t_devices)+\
                  '_seed'+str(args.seed),'wb') as f:
            pk.dump(d2dset,f)
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

# labels assignment and populate device datasets
labels = 10 #all three datasets have three labels
d_dsets = {} #these are indexes
if args.dset_split < 2: 
    d_train_ls = {i:np.where(d_train.targets==i)[0] for i in range(labels)}
elif args.dset_split == 2:
    d_train_ls0 = {i:np.where(d0.targets==i)[0] for i in range(labels)}
    d_train_ls1 = {i:np.where(d1.targets==i)[0] for i in range(labels)}

if args.label_split == 1:
    # determination of the device datasets requires
    if args.labels_type == 'mild':
        if 'MM' in args.dset_type and args.dset_split == 0:
            # lpd = [random.sample(range(labels),6) for i in range(args.t_devices)]
            lpd = [random.sample(range(labels),8) for i in range(args.l_devices)]
            lpd_u = [lpd[random.sample(range(args.l_devices),1)[0]] for i in range(args.u_devices)]
            lpd += lpd_u
            print(lpd)
            # td_qty = np.round(np.random.dirichlet(5*np.ones(6),args.t_devices),2)
            td_qty = np.round(np.random.dirichlet(5*np.ones(8),args.t_devices),2)
        elif 'MM' in args.split_type and args.dset_split == 1:
            lpd = [random.sample(range(labels),6) for i in range(args.t_devices)]
            td_qty = np.round(np.random.dirichlet(5*np.ones(6),args.t_devices),2)
        elif args.dset_split == 2: #'MM' in args.split_type and 
            lpd = [random.sample(range(labels),6) for i in range(args.t_devices)]
            td_qty = np.round(np.random.dirichlet(5*np.ones(6),args.t_devices),2)            
        else:
            # lpd = [random.sample(range(labels),3) for i in range(args.t_devices)]
            lpd = [random.sample(range(labels),4) for i in range(args.l_devices)]
            lpd_u = [lpd[random.sample(range(args.l_devices),1)[0]] for i in range(args.u_devices)]
            lpd += lpd_u
            print(lpd)
            td_qty = np.round(np.random.dirichlet(5*np.ones(4),args.t_devices),2)
        for trow in td_qty:
            if np.round(sum(trow),2) < 1:
                ind_min = np.argmin(trow)
                trow[ind_min] += 1 - np.round(sum(trow),2)
            elif np.round(sum(trow),2) > 1:
                ind_max = np.argmax(trow)
                trow[ind_max] -= np.round(sum(trow),2) - 1
        
        d_dset_sqtys = np.array([alld_qty[i]*td_qty[i,:] for i in range(args.t_devices)])
        d_dset_sqtys = np.round(d_dset_sqtys,0).astype(int)
    elif args.labels_type == 'extreme':
        lpd = [random.sample(range(labels),1) for i in range(args.t_devices)]
        d_dset_sqtys = [np.random.multinomial(alld_qty[i],[1]) \
                         for i in range(args.t_devices)]
    else:
        raise TypeError('labels type invalid')
elif args.label_split == 0: #i.e., iid
    lpd = [list(range(10)) for i in range(args.t_devices)]
    d_dset_sqtys = [np.random.multinomial(alld_qty[i],[np.round(1/10,3)]*10) \
                     for i in range(args.t_devices)]    
else:
    raise TypeError('label split invalid')

for i in range(args.t_devices): 
    d_dsets[i] = []
    c_labels = lpd[i]
    # d2dset[i]
    for ti,tj in enumerate(d_dset_sqtys[i]):
        if args.dset_split != 2:
            if tj > len(d_train_ls[c_labels[ti]].tolist()):
                td_dset = deepcopy(d_train_ls[c_labels[ti]])
                while len(td_dset) < tj:
                    if len(td_dset)-tj > len(d_train_ls[c_labels[ti]].tolist()):
                        td_dset += td_dset
                    else:
                        td_dset = list(td_dset)
                        td_dset += random.sample(d_train_ls[c_labels[ti]].tolist(),tj-len(td_dset))
                random.shuffle(td_dset)
            else:
                td_dset = random.sample(d_train_ls[c_labels[ti]].tolist(),tj)
            d_dsets[i].extend(td_dset)
        elif args.dset_split == 2:
            if d2dset[i] == 0:
                c_dtrain_ls = d_train_ls0
            elif d2dset[i] == 1:
                c_dtrain_ls = d_train_ls1
            
            if tj > len(c_dtrain_ls[c_labels[ti]].tolist()):
                td_dset = deepcopy(c_dtrain_ls[c_labels[ti]])
                while len(td_dset) < tj:
                    if len(td_dset)-tj > len(c_dtrain_ls[c_labels[ti]].tolist()):
                        td_dset += td_dset
                    else:
                        td_dset = list(td_dset)
                        td_dset += random.sample(c_dtrain_ls[c_labels[ti]].tolist(),tj-len(td_dset))
                random.shuffle(td_dset)
            else:
                td_dset = random.sample(c_dtrain_ls[c_labels[ti]].tolist(),tj)
            d_dsets[i].extend(td_dset)

if args.dset_split == 0:
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.dset_type+'_'+args.labels_type+'_lpd','wb') as f:
        pk.dump(lpd,f)
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.dset_type+'_'+args.labels_type+'_dindexsets','wb') as f:
        pk.dump(d_dsets,f)
elif args.dset_split == 1:
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.split_type+'_'+args.labels_type+'_lpd','wb') as f:
        pk.dump(lpd,f)
    with open(cwd+'/data_div/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.split_type+'_'+args.labels_type+'_dindexsets','wb') as f:
        pk.dump(d_dsets,f)
elif args.dset_split == 2:
    with open(cwd+'/data_div/total_devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.split_type+'_'+args.labels_type+'_lpd','wb') as f:
        pk.dump(lpd,f)
    with open(cwd+'/data_div/total_devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.split_type+'_'+args.labels_type+'_dindexsets','wb') as f:
        pk.dump(d_dsets,f)

# input('a')

# %% source target label re-assignment func
def st_relab(s_dset,t_dset,d_train,d_t2=None):
    if d_t2 == None:
        ## populate the set
        sl_dset = [(d_train[td][0],0) for td in s_dset]
        tl_dset = [(d_train[td][0],1) for td in t_dset]
    else:
        sl_dset = [(d_train[td][0],0) for td in s_dset]
        tl_dset = [(d_t2[td][0],1) for td in t_dset]        
    return sl_dset, tl_dset

# %% single training iter func
def div_roi(loc_model,bs,lr,l_dset=None,st=None,\
            device=device,dt=None):
    tobj = LocalUpdate(device,bs=bs,lr=lr,epochs=1,st=st,\
            dataset=dt,indexes=l_dset)
    _,w,loss = tobj.train(net=loc_model.to(device))
    return w,loss

# %% setup training vars
if args.div_nn == 'MLP':
    if args.dset_split < 2: 
        d_in = np.prod(d_train[0][0].shape)
    elif args.dset_split == 2:
        d_in = np.prod(d0[0][0].shape)
    d_h = 64
    d_out = 2
    start_net = MLP(d_in,d_h,d_out).to(device)
    os_append = 'MLP_start_w'
elif args.div_nn == 'CNN':
    nchannels = 1
    nclasses = 2
    start_net = CNN(nchannels,nclasses).to(device)
    os_append = 'CNN_start_w'
elif args.div_nn == 'GCNN':
    nchannels = 1
    nclasses = 2 
    start_net = GCNN(nchannels,nclasses).to(device)
    os_append = 'GCNN_start_w'
try:
    with open(cwd+'/div_utils/{}'.format(os_append),'rb') as f:
        start_w = pk.load(f)
    start_net.load_state_dict(start_w)
except:
    start_w = start_net.state_dict()
    with open(cwd+'/div_utils/{}'.format(os_append),'wb') as f:
        pk.dump(start_w,f)
print(start_net)

# %% pairwise training loops
lab2ulab_accs = np.zeros(shape=(args.t_devices,args.t_devices)).astype(int) #labeled to unlabeled accuracies

def nearest_batch(index_set,bs_size=args.div_bs):
    return int(np.ceil(len(index_set)*0.36/bs_size)*bs_size)

for i in range(args.t_devices):
    for j in range(args.t_devices):
        if lab2ulab_accs[i,j] == 0 and i != j:
            # get the relabelled datasets
            if args.dset_split < 2: 
                sl_dset,tl_dset = st_relab(d_dsets[i],d_dsets[j],d_train)
            elif args.dset_split == 2:
                sl_dset,tl_dset = st_relab(d_dsets[i],d_dsets[j],d_train_dict[i],d_train_dict[j])
            
            st_net = deepcopy(start_net)
            
            # training, combining, and testing loop
            for tc in range(args.div_ttime):
                # one training iteration
                s_temp_set = random.sample(d_dsets[i],\
                        nearest_batch(d_dsets[i]))
                t_temp_set = random.sample(d_dsets[j],\
                        nearest_batch(d_dsets[j]))

                if args.dset_split < 2:                    
                    s_w,s_loss = div_roi(deepcopy(st_net),st='source',\
                            bs=args.div_bs,lr=args.div_lr,\
                            l_dset=s_temp_set,dt=d_train)
                    t_w,t_loss = div_roi(deepcopy(st_net),st='target',\
                            bs=args.div_bs,lr=args.div_lr,\
                            l_dset=t_temp_set,dt=d_train)
                elif args.dset_split == 2:
                    s_w,s_loss = div_roi(deepcopy(st_net),st='source',\
                            bs=args.div_bs,lr=args.div_lr,\
                            l_dset=s_temp_set,dt=d_train_dict[i])
                    t_w,t_loss = div_roi(deepcopy(st_net),st='target',\
                            bs=args.div_bs,lr=args.div_lr,\
                            l_dset=t_temp_set,dt=d_train_dict[j])
                # perform unweighted avg for the two devices
                w_avg = wAvg([s_w,t_w])
                st_net.load_state_dict(w_avg)

            # calc acc/error - done in distributed way
            if args.dset_split < 2: 
                s_acc,s_loss = test_img(st_net,args.div_bs,dset=d_train,\
                         indx=random.sample(d_dsets[i],10*args.div_bs),\
                         st='source',device=device)
                t_acc,t_loss = test_img(st_net,args.div_bs,dset=d_train,\
                         indx=random.sample(d_dsets[j],10*args.div_bs),\
                         st='target',device=device)
            elif args.dset_split == 2:
                s_acc,s_loss = test_img(st_net,args.div_bs,dset=d_train_dict[i],\
                         indx=random.sample(d_dsets[i],10*args.div_bs),\
                         st='source',device=device)
                t_acc,t_loss = test_img(st_net,args.div_bs,dset=d_train_dict[j],\
                         indx=random.sample(d_dsets[j],10*args.div_bs),\
                         st='target',device=device)
            ovr_acc = (s_acc+t_acc)/2
            lab2ulab_accs[i,j] = ovr_acc
            try:
                lab2ulab_accs[j,i] = ovr_acc
            except:
                fill_var = 1

# %% save the results
if args.dset_split == 0:
    if 'MM' in args.dset_type:
        end = '_base_6'
    else:
        end = ''
    with open(cwd+'/div_results/div_vals/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
        +'_'+args.div_nn\
        +'_'+args.dset_type+'_'+args.labels_type+end,'wb') as f:
        pk.dump(lab2ulab_accs,f)
else:
    if 'MM' in args.split_type:
        end = '_base_6'
    else:
        end = ''
    if args.dset_split == 1:
        with open(cwd+'/div_results/div_vals/devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.split_type+'_'+args.labels_type+end,'wb') as f:
            pk.dump(lab2ulab_accs,f)
    elif args.dset_split == 2:
        with open(cwd+'/div_results/div_vals/total_devices'+str(args.t_devices)+'_seed'+str(args.seed)\
            +'_'+args.div_nn\
            +'_'+args.split_type+'_'+args.labels_type+end,'wb') as f:
            pk.dump(lab2ulab_accs,f)