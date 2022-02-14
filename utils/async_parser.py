# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:42:22 2021

@author: henry
"""
import argparse

# %% parser function
def ml_parser():
    parser = argparse.ArgumentParser()    
    # ml infrastructure settings 
    parser.add_argument('--comp',type=str,default='gpu',\
                        choices=['gpu','cpu'],\
                        help='gpu or cpu')
    parser.add_argument('--gpu_num',type=str,default='0',\
                        help='which gpu to use')
    parser.add_argument('--train_time',type=int,default=60,\
                        help='total time of the system - epochs') #prev default 100
    parser.add_argument('--bs_stat',type=str,default='small',\
                        help='large or small batch size')
        
    # parser
    args = parser.parse_args()
    return args

