# -*- coding: utf-8 -*-
import argparse

# %% parser function
def optim_parser():
    parser = argparse.ArgumentParser()
    ## code system settings
    parser.add_argument('--seed',type=int,default=1) #[1,2,3,4,5]
    parser.add_argument('--optim_save',type=bool,default=False,\
                        help='save the run data or not')
    
    ## devices and their characteristics
    parser.add_argument('--t_devices',type=int,default=10,\
                        help='total_networked_devices') #10
    parser.add_argument('--l_devices',type=int,default=5,\
                        help='devices w labelled data') #5
    # 'devices w/out labelled data' - see bottom
    
    # 1000, 800, 200
    parser.add_argument('--avg_uqty',type=int,default=1000,\
                        help='avg data qty at fully unlabeled devices')
    parser.add_argument('--avg_lqty_l',type=int,default=1600,\
                        help='avg labelled qty at labelled devices')
    parser.add_argument('--avg_lqty_u',type=int,default=200,\
                        help='avg unlabelled qty at labelled devices') 
    parser.add_argument('--avg_size',type=str,default='small')
    
    ## scaling
    parser.add_argument('--phi_s',type=float,default=1,\
                        help='scaling the source errors') #0.5
    parser.add_argument('--phi_t',type=float,default=50,\
                        help='scaling the target errors') #50 #0.5
    parser.add_argument('--phi_e',type=float,default=1e0,\
                        help='scaling the energy term') #1e1
    
    ## optimization constants
    parser.add_argument('--approx_iters',type=int,default=50,\
                        help='posynomial approximation iterations') #50
    parser.add_argument('--l_delta',type=float,default=1e-3,\
                        help='probabiliy 1-\delta term, accuracy metric') #1e-2
    
    ## ablation variables 
    parser.add_argument('--init_test',type=int,default=0,\
                        help='initial convergence tests')
    parser.add_argument('--div_flag',type=int,default=1,\
                        help='estimate divergence yes or no') #1 
    ## divergence estimation variables
    # div est infrastructure vars
    parser.add_argument('--div_comp',type=str,default='gpu',\
                        choices=['cpu','gpu'])
    parser.add_argument('--div_gpu_num',type=int,default=0,\
                        help='based on your devices')
    parser.add_argument('--div_ttime',type=int,default=10,\
                        help='divergence estimation total iteration loops')        
    parser.add_argument('--div_nn',type=str,default='CNN',\
                        choices=['MLP','CNN','GCNN'],\
                        help='neural network for divergence estimation')
    parser.add_argument('--div_lr',type=float,default=1e-2)    
    parser.add_argument('--div_bs',type=int,default=10)
    
    # div est data + label vars 
    parser.add_argument('--dset_split',type=int,default=0,\
                        help='whether there are multiple datasets'+\
                        '0:single dataset, 1: mixed dataset on device, 2: mixed '+\
                        'datasets across network',\
                        choices=[0,1,2])
    parser.add_argument('--split_type',type=str,default='M+U',\
                        choices=['M+MM','M+U','M+S','MM+U','MM+S','S+U','A'],
                        help='{M+S:mnist+svhn,'+\
                        'M+U:mnist+usps,S+U:svhn+usps,A:all}')
    parser.add_argument('--dset_type',type=str,default='M',\
                        choices=['M','S','U','MM'],\
                        help='{M:mnist,S:svhn,U:usps,MM:mnist-m}')
    
    parser.add_argument('--label_split',type=int,default=1,\
                        help='true:1 or false:0 (false = iid)') # 1
    parser.add_argument('--labels_type',type=str,default='mild',\
                        choices=['mild','extreme'],\
                        help='type of labels assignment')
    
    # source training variables
    parser.add_argument('--st_time',type=int,default=100,\
                        help='source training time') #100
    parser.add_argument('--grad_rev',type=bool,default=False,\
                        help='gradient reversal layer, yes/no')
    parser.add_argument('--fl',type=bool,default=True,\
                        help='FL style pre-training')

    # nrg variables
    parser.add_argument('--p2bits',type=int,default=1e9,\
                        help='model 2 bits')
    parser.add_argument('--nrg_mt',type=int,default=1,\
                        help='run energy compute for model transfers')
    # parser
    args = parser.parse_args()
    
    # remaining calcs
    args.u_devices = args.t_devices - args.l_devices
    return args

# %% argparser for the divergence ablation study only
def div_ablate_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--div_ablation',type=int,default=1,\
                        help='Tests for divergence ablation flag')
    parser.add_argument('--div_hetero',type=str,default='random',\
                        choices=['random','extreme','estimated'])
    args = parser.parse_args()
    return args
    
    
    