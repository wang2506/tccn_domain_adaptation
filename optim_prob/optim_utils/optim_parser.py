# -*- coding: utf-8 -*-
"""
@author: ch5b2
"""

import argparse

# %% parser function
def optim_parser():
    parser = argparse.ArgumentParser()
    
    ## devices and their characteristics    
    parser.add_argument('--t_devices',type=int,default=10,\
                        help='total_networked_devices')
    parser.add_argument('--l_devices',type=int,default=5,\
                        help='devices w labelled data')
    # 'devices w/out labelled data' - see bottom
    
    parser.add_argument('--avg_uqty',type=int,default=1000,\
                        help='avg data qty at fully unlabeled devices')
    parser.add_argument('--avg_lqty_l',type=int,default=800,\
                        help='avg labelled qty at labelled devices')
    parser.add_argument('--avg_lqty_u',type=int,default=200,\
                        help='avg unlabelled qty at labelled devices')


    ## optimization constants
    parser.add_argument('--approx_iters',type=int,default=5,\
                        help='posynomial approximation iterations')
    parser.add_argument('--l_delta',type=float,default=1e-2,\
                        help='probabiliy 1-\delta term, accuracy metric')
    # parser.add_argument('--params_dim',type=int,default=1e3,\
    #                     help='dimension of hypothesis space')
    
    
    # parser.add_argument('--gtime',type=int,default=10,\
    #                     help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
        
    parser.add_argument('--save',type=bool,default=False,\
                        help='save the run data or not')
    
    ## ablation variables
    parser.add_argument('--div_flag',type=bool,default=False,\
                        help='estimate divergence yes or no')
    
        
    # parser
    args = parser.parse_args()
    
    # remaining calcs
    args.u_devices = args.t_devices - args.l_devices
    return args

# %% temp

    # parser.add_argument('--data_style',type=str,default='FMNIST',\
    #                     choices=['FMNIST','MNIST','CIFAR10'])
    # parser.add_argument('--iid_style',type=str,default='mild',\
    #                     choices=['extreme','mild','iid'],\
    #                     help='noniid/iid styles')