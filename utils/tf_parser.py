# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:42:22 2021

@author: henry
"""
import argparse

# %% parser function
def tf_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_style',type=str,default='mnist',\
                        choices=['fmnist','mnist'],\
                        help='data style fashion-mnist')
    parser.add_argument('--nn_style',type=str,default='CNN',\
                        choices=['CNN','MLP','VGG','FCNN'],\
                        help='neural network style: cnn/mlp/vgg11')
    parser.add_argument('--iid_style',type=str,default='iid',\
                        choices=['iid','mild','extreme'],\
                        help='non-iid severity')
    
    parser.add_argument('--comp',type=str,default='cpu',\
                        choices=['gpu','cpu'],\
                        help='gpu or cpu')
    parser.add_argument('--gpu_num',type=str,default='0',\
                        help='which gpu to use')
    

    
    parser.add_argument('--avg_qty',type=int,default='1000',\
                        help='data per device') #3000
    parser.add_argument('--static',type=bool,default=True)
    parser.add_argument('--time',type=int,default=60,\
                        help='total time of the system - epochs') #prev default 100    
    
    parser.add_argument('--seed',type=int, default=1,\
                        help='Seeding for RNG')
    
    parser.add_argument('--domains',type=str,default='binary',\
                        choices=['binary','multi'])
    # not sure if multi makes sense...unless you want some of the non-invariant stuff
    # if replace sigmoid with softmax...we'll see - may need some data comparisons
    
    
    parser.add_argument('--lr',type=float,default=1e-3,\
                        help='learning rate')    
    parser.add_argument('--bs',type=int,default=10)
    # parser.add_argument('--bs_stat',type=str,default='small',\
    #                     help='large or small batch size')    
    # parser.add_argument('--')
    
    # parser
    args = parser.parse_args()
    return args












