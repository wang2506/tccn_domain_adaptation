# -*- coding: utf-8 -*-
"""
@author: ch5b2
"""

import os
import cvxpy as cp
import numpy as np
import pickle as pk
import random

from optim_utils.optim_parser import optim_parser

cwd = os.getcwd()
args = optim_parser()

np.random.seed(args.seed)
random.seed(args.seed)


# %% variable declarations
phi_s = 0.25 # source errors
phi_t = 0.5 # target errors
phi_e = 0.25 # energy consumptions


psi = cp.Variable(args.t_devices,pos=True)#,boolean=True)
alpha = cp.Variable((args.t_devices,args.t_devices),pos=True)


# %% shared entities
## data qty
all_u_qtys = np.random.normal(args.avg_uqty,args.avg_uqty/6,\
            size=args.u_devices).astype(int) #all_unlabelled_qtys

split_lqtys = np.random.normal(args.avg_lqty_l,args.avg_lqty_l/6,\
            size=args.l_devices).astype(int)
split_uqtys = np.random.normal(args.avg_lqty_u,args.avg_lqty_u/6,\
            size=args.l_devices).astype(int)

print(all_u_qtys)
print(split_lqtys)
print(split_uqtys)

net_l_qtys = split_lqtys + split_uqtys

data_qty_alld = list(net_l_qtys)+list(all_u_qtys)

## epsilon hat - average loss at the devices with labelled data, as measured on their labelled
hat_ep = [] #sequentially stored

min_ep_vect = 1e-3
max_ep_vect = 5e-1
temp_ep_vect = (min_ep_vect+(max_ep_vect-min_ep_vect)*np.random.rand(args.t_devices)).tolist()

# temporarily assign constant values
# later, we will need to figure out a way to estimate them
for i in range(args.l_devices):
    t_hat_ep = random.sample(temp_ep_vect,1)
    hat_ep.extend(t_hat_ep)

## ordering devices (labelled and unlabelled combined)
device_order = []
# # init_dlist = list(np.arange(0,10,1))
# for i in range(args.t_devices): 
#     device_order.append()
    
# for now, just sequentially, all labelled, then unlabelled
device_order = list(np.arange(0,args.l_devices+args.u_devices,1))

hat_ep_alld = []
for i in range(args.t_devices):
    if i < args.l_devices:
        hat_ep_alld.append(hat_ep[i])
    else:
        hat_ep_alld.append(1e2)

## rademacher estimates
rad_s = [np.sqrt(2*np.log(net_l_qtys[i])/net_l_qtys[i]) for i in range(args.l_devices)]
rad_t = [np.sqrt(2*np.log(all_u_qtys[i])/all_u_qtys[i]) for i in range(args.u_devices)]

rad_alld = rad_s+rad_t

## sqrt of log term
sqrt_s = [3*np.sqrt(np.log(2/args.l_delta)/(2*net_l_qtys[i])) for i in range(args.l_devices)]
sqrt_t = [3*np.sqrt(np.log(2/args.l_delta)/(2*all_u_qtys[i])) for i in range(args.u_devices)]

sqrt_alld = sqrt_s+sqrt_t

# %% objective fxn - term 1 [source error]
def s_err_calc(psi,rads=rad_alld,sqrts=sqrt_alld,hat_ep=hat_ep_alld,args=args):
    s_err = 1e-6
    for i in range(args.t_devices):
        s_err += (1-psi[i])*(hat_ep[i]+2*rads[i]+sqrts[i])
        
    return s_err

s_err = s_err_calc(psi)

# needs a posynomial approximation
chi = cp.Variable(args.t_devices,pos=True)

psi_init = 0.5*np.ones(args.t_devices)


# %% objective fxn - term 2 [target error]
t_err = 1e-6
for j in range(args.t_devices):
    temp_calc = 1e-6
    for i in range(args.t_devices):
        if args.div_flag == False:
            temp_calc += (1-psi[i])*alpha[i,j]*\
                (hat_ep_alld[i]+2*rad_alld[i]+sqrt_alld[i]+\
                4*rad_alld[i]+4*rad_alld[j]+6*sqrt_alld[i]+6*sqrt_alld[j])
        else:
            raise TypeError('no work atm')
        
    t_err += psi[j]*temp_calc


# %% objective fxn - term 3 [energy costs]
e_err = 1e-6


# %% constraints
constraints = []

con_alpha = []
for i in range(args.t_devices):
    for j in range(args.t_devices):
        con_alpha.append(alpha[i,j] <= 1)
constraints.extend(con_alpha)


# %% posynomial approximation fxn
def posy_init(iter_num,cp_vars,args=args):
    ## iter_num - iteration number
    ## cp_vars - dict of optimization variables on which to perform posynomial approximation
    
    if iter_num == 0:
        psi_init = 0.5*np.ones(args.t_devices)
        chi_init = 10*np.ones(args.t_devices)

    else:
        psi_init = cp_vars['psi'].value
        chi_init = cp_vars['chi'].value

    return psi_init,chi_init

# %% combine and run

for c_iter in range(args.approx_iters):
    t_dict = {'psi':psi,'chi':chi}
    psi_init,chi_init = posy_init(c_iter,t_dict)
    
    t_sum = phi_s*s_err + phi_t*t_err
    
    # prob = cp.Problem(cp.Minimize(t_err),constraints=constraints)
    prob = cp.Problem(cp.Minimize(s_err))
    prob.solve(solver=cp.MOSEK,gp=True)#,verbose=True)















# %% backups
## no initial constant for s_err
# for i in range(args.t_devices):
#     if i == 0:
#         s_err = (1-psi[i])*(hat_ep_alld[i]+2*rad_alld[i]+sqrt_alld[i])
#     else:
#         s_err += (1-psi[i])*(hat_ep_alld[i]+2*rad_alld[i]+sqrt_alld[i])



## no initial constant for t_err
# for j in range(args.t_devices):
#     for i in range(args.t_devices):
#         if args.div_flag == False:
#             if i == 0:
#                 temp_calc = psi[i]*alpha[i,j]*\
#                     (hat_ep_alld[i]+2*rad_alld[i]+sqrt_alld[i]+\
#                     4*rad_alld[i]+4*rad_alld[j]+6*sqrt_alld[i]+6*sqrt_alld[j])
#             else:
#                 temp_calc += psi[i]*alpha[i,j]*\
#                     (hat_ep_alld[i]+2*rad_alld[i]+sqrt_alld[i]+\
#                     4*rad_alld[i]+4*rad_alld[j]+6*sqrt_alld[i]+6*sqrt_alld[j])
#         else:
            
            
    
#     if j == 0:
#         t_err = 0
#     else:
#         t_err += 0







