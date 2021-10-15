# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:42:22 2021

@author: henry
"""
import argparse

# %% parser function
def optim_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=5,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=10,\
                        help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
    
    parser.add_argument('--max_agg_p',type=int,default=25)
    parser.add_argument('--min_agg_p',type=int,default=1)
    
    parser.add_argument('--tml',type=int,default=5e3)
    #default = 1e2 works well, try others 
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=50,\
                        help='total num of approximation iterations')# default = 50
        
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    # parser
    args = parser.parse_args()
    return args

# %% 

def optim_parser_obj():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=20,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=2,\
                        help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
    
    parser.add_argument('--max_agg_p',type=int,default=25)
    parser.add_argument('--min_agg_p',type=int,default=1)
    
    parser.add_argument('--tml',type=int,default=1e2)
    #default = 1e2 works well, try others 
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=50,\
                        help='total num of approximation iterations')# default = 50
        
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    # parser
    args = parser.parse_args()
    return args

# %%
def optim_parser_zeta2():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=5,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=2,\
                        help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
    
    parser.add_argument('--max_agg_p',type=int,default=25)
    parser.add_argument('--min_agg_p',type=int,default=1)
    
    parser.add_argument('--tml',type=int,default=1e2)
    #default = 1e2 works well, try others 
    
    parser.add_argument('--zeta2',type=float,default=1e1) #1e-1,1e-3,1e-5
    #afterwards, also ran 1e-2, 1e-4
    #with changes, now ran 1e1, 1e0 as well
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=50,\
                        help='total num of approximation iterations')# default = 50
        
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    # parser
    args = parser.parse_args()
    return args

# %% 
def optim_parser_minib():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=5,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=2,\
                        help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
    
    parser.add_argument('--max_agg_p',type=int,default=25)
    parser.add_argument('--min_agg_p',type=int,default=1)
    
    parser.add_argument('--tml',type=int,default=1e2)
    #default = 1e2 works well, try others 
    
    parser.add_argument('--zeta2',type=float,default=1e-3) #1e-1,1e-3,1e-5
    #afterwards, also ran 1e-2, 1e-4
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=50,\
                        help='total num of approximation iterations')# default = 50
    
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    
    # c1,c2,c3 parameters for the optimization
    # corresponding to alpha [energy], beta [delay], and gamma [ML]
    parser.add_argument('--c1',type=float,default=1e4/1e2)  # on 2.5, need2run
    ## rerun the entire system 
    # 1e1/1e2, 1e2/1e2, 1e3/1e2 - test these three first 
    # 5e0/1e2, 5e1/1e2, 5e2/1e2, 5e3/1e2, 1e4/1e2
    # 2.5e1/1e2, 2.5e2/1e2
    # c1_vals = [5e0/1e2,1e1/1e2,2.5e1/1e2,5e1/1e2,\
    #             1e2/1e2,2.5e2/1e2,5e2/1e2,1e3/1e2,\
    #             5e3/1e2,1e4/1e2]    
    
    # parser
    args = parser.parse_args()
    return args

# %% 
def optim_parser_objK():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=10,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=2,\
                        help='total num of global iterations') #default=2
    # 2 to 7
    
    parser.add_argument('--seed',type=int,default=1) 
    
    parser.add_argument('--max_agg_p',type=int,default=25) 
    parser.add_argument('--min_agg_p',type=int,default=1) 
    
    parser.add_argument('--tml',type=int,default=3.2e5) 
    # 1e2 - 3
    # 5e2 - 3
    # 8e2 - 3
    # 9e2 - 3
    # 1e3 - 3
    # 2e3 - 4
    # 5e3 - 4
    # 1e4 - 4
    # 5e4 - 4
    # 8e4 - 4
    # 9e4 - 5
    # 1e5 - 5
    # 2e5 - 6
    # 2.5e5 - 6
    
    # 3e5 - 6
    # 3.1e5 - 6 *
    # 3.2e5 - 7 *
    # 3.5e5 - 7
    # 4e5 - 7
    # 4.6e5 - 7 * 
    # 4.7e5 - 8 *
    # 4.8e5 - 8
    
    # 5e5 - 8
    # 6e5 - 8
    # 6.5e5 - 8 
    # 7e5 - 8
    # 8e5 - 8
    # 1e6 - geq 8
    # 2e6 - geq 8
    
    parser.add_argument('--zeta2',type=float,default=1e-5) #1e-3) #1e-1,1e-3,1e-5
    #afterwards, also ran 1e-2, 1e-4
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=20,\
                        help='total num of approximation iterations')# default = 50
    
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    
    # c1,c2,c3 parameters for the optimization
    # corresponding to alpha [energy], beta [delay], and gamma [ML]
    parser.add_argument('--c1',type=float,default=1e0)
    
    # parser
    args = parser.parse_args()
    return args


# %% 
def optim_parser_c1():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=5,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=2,\
                        help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
    
    parser.add_argument('--max_agg_p',type=int,default=25)
    parser.add_argument('--min_agg_p',type=int,default=1)
    
    parser.add_argument('--tml',type=int,default=1e2)
    #default = 1e2 works well, try others 
    
    parser.add_argument('--zeta2',type=float,default=1e-5) #1e-1,1e-3,1e-5
    #afterwards, also ran 1e-2, 1e-4
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=200,\
                        help='total num of approximation iterations')# default = 50
    
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    
    # c1,c2,c3 parameters for the optimization
    # corresponding to alpha [energy], beta [delay], and gamma [ML]
    parser.add_argument('--c1',type=float,default=1e4/1e2) #1e1,1e2,1e3,1e4
    parser.add_argument('--c2',type=float,default=1e2/1e2) #1e1,1e2,1e3
    parser.add_argument('--c3',type=float,default=1e1/1e2)
    # parser
    args = parser.parse_args()
    return args

# %% 
def optim_parser_c2():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=5,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=3,\
                        help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
    
    parser.add_argument('--max_agg_p',type=int,default=25)
    parser.add_argument('--min_agg_p',type=int,default=1)
    
    parser.add_argument('--tml',type=int,default=1e3)
    #default = 1e2 works well, try others 
    
    parser.add_argument('--zeta2',type=float,default=1e-5) #1e-1,1e-3,1e-5
    #afterwards, also ran 1e-2, 1e-4
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=50,\
                        help='total num of approximation iterations')# default = 50
    
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    
    # c1,c2,c3 parameters for the optimization
    # corresponding to alpha [energy], beta [delay], and gamma [ML]
    parser.add_argument('--c1',type=float,default=1e2/1e2)
    parser.add_argument('--c2',type=float,default=1e5)
    # 4.735, 4.733, 4.737
    # 1e0, 1e1, 1e2, 1e3, 1e4
    # 5e1, 7.5e1
    # 2e1, 2.5e1
    # 4e1, 3e1, 4.5e1
    # 4.6e1, 4.7e1, 4.8e1, 4.9e1
    # 4.1e1, 4.2e1, 4.3e1, 4.4e1
    
    parser.add_argument('--c3',type=float,default=1e2/1e2)
    # parser
    args = parser.parse_args()
    return args


# %% 
def optim_parser_c3():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--devices',type=int,default=5,\
                        help='networked_devices') #default = 5
    parser.add_argument('--avg_qty',type=int,default=3000,\
                        help='avg data qty at a device used to determine '+\
                        'data variance')
    
    parser.add_argument('--data_style',type=str,default='FMNIST',\
                        choices=['FMNIST','MNIST','CIFAR10'])
    parser.add_argument('--iid_style',type=str,default='mild',\
                        choices=['extreme','mild','iid'],\
                        help='noniid/iid styles')

    parser.add_argument('--gtime',type=int,default=2,\
                        help='total num of global iterations') #default=2
    
    parser.add_argument('--seed',type=int,default=1)
    
    parser.add_argument('--max_agg_p',type=int,default=25)
    parser.add_argument('--min_agg_p',type=int,default=1)
    
    parser.add_argument('--tml',type=int,default=1e3)
    #default = 1e2 works well, try others 
    
    parser.add_argument('--zeta2',type=float,default=1e-5) #1e-1,1e-3,1e-5
    #afterwards, also ran 1e-2, 1e-4
    
    parser.add_argument('--bs_stat',type=str,default='large',\
                        help='large or small batch size')
    
    parser.add_argument('--max_approx_iters',type=int,default=200,\
                        help='total num of approximation iterations')# default = 50
    
    parser.add_argument('--save',type=bool,default=True,\
                        help='save the run data or not')
    
    
    # c1,c2,c3 parameters for the optimization
    # corresponding to alpha [energy], beta [delay], and gamma [ML]
    parser.add_argument('--c1',type=float,default=1e1) #1e0
    parser.add_argument('--c2',type=float,default=1e1)
    parser.add_argument('--c3',type=float,default=4e-3)
    # 5e-5, 5e-4
    # 1e-4, 1e-3
    # 5e-3, 5e-2, 5e-1, 5e0
    # 1e-2, 1e-1, 1e0, 1e1
    # in [5e-4, 1e-3]
    # 6,7,8,9
    # in [1e-3, 5e-3]
    # 2,3,4
    # parser
    args = parser.parse_args()
    return args

























