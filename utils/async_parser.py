# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:42:22 2021

@author: henry
"""
import argparse

# %% parser function
def ml_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_style',type=str,default='fmnist',\
                        choices=['fmnist','mnist'],\
                        help='data style fashion-mnist')
    parser.add_argument('--nn_style',type=str,default='CNN',\
                        choices=['CNN','MLP','VGG','FCNN'],\
                        help='neural network style: cnn/mlp/vgg11')
    
    parser.add_argument('--devices',type=int,default=10,\
                        help='networked_devices') #20
    
    parser.add_argument('--hf_d',type=int,default=1,\
                        help='Number of devices that aggregate every iteration '+\
                        '_hf_d: high freq devices') # 2 
    parser.add_argument('--lf_d',type=int,default=1,\
                        help='low_freq devices') #4 '; # devices that agg at max')
    
    parser.add_argument('--max_agg_p',type=int,default=25,\
                        help='max aggregation period') #previously 10
    parser.add_argument('--min_agg_p',type=int,default=1,\
                        help='min agg period')
    
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')
    
    parser.add_argument('--comp',type=str,default='cpu',\
                        choices=['gpu','cpu'],\
                        help='gpu or cpu')
    
    parser.add_argument('--gpu_num',type=str,default='0',\
                        help='which gpu to use')
    
    parser.add_argument('--time',type=int,default=60,\
                        help='total time of the system - epochs') #prev default 100
    
    parser.add_argument('--max_lr',type=float,default=1e-2+1e-4,\
                        help='max learning rate')
    parser.add_argument('--min_lr',type=float,default=1e-4,\
                        help='min_learning rate')
    
    parser.add_argument('--bs_stat',type=str,default='small',\
                        help='large or small batch size')
    
    parser.add_argument('--avg_qty',type=int,default='1000',\
                        help='data per device') #3000
    
    parser.add_argument('--static',type=bool,default=True)
    
    # parser
    args = parser.parse_args()
    return args


# %%
# [31, 32, 34, 33, 30, 36, 30, 37, 32, 40, 38, 33, 30, 38, 34, 32, 39, 30, 33, 38]
# [10, 9, 1, 5, 3, 4, 5, 10, 5, 4, 7, 8, 4, 4, 7, 1, 4, 10, 10, 4]
