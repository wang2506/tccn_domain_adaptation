# -*- coding: utf-8 -*-
"""
@author: ch5b2
"""

import os
import cvxpy as cp
import numpy as np
import pickle as pk
import random
from copy import deepcopy


from optim_utils.optim_parser import optim_parser

cwd = os.getcwd()
args = optim_parser()

np.random.seed(args.seed)
random.seed(args.seed)

# %% variable declarations
phi_s = args.phi_s # source errors
phi_t = args.phi_t # target errors
phi_e = args.phi_e # energy consumptions

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
temp_ep_vect = (min_ep_vect+(max_ep_vect-min_ep_vect) \
                *np.random.rand(args.t_devices)).tolist()

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

## empirical hypothesis mismatch error
ep_mismatch = {}
min_mismatch = 1e-3
max_mismatch = 5e-1

for i in range(args.t_devices):
    temp_ep_mismatch = (min_mismatch+(max_mismatch-min_mismatch)\
                        *np.random.rand(args.t_devices)).tolist()
    ep_mismatch[i] = []
    for j in range(args.t_devices):
        ep_mismatch[i].extend(random.sample(temp_ep_mismatch,1))

## divergence terms


## rademacher estimates
rad_s = [np.sqrt(2*np.log(net_l_qtys[i])/net_l_qtys[i]) for i in range(args.l_devices)]
rad_t = [np.sqrt(2*np.log(all_u_qtys[i])/all_u_qtys[i]) for i in range(args.u_devices)]

rad_alld = rad_s+rad_t

## sqrt of log term
sqrt_s = [3*np.sqrt(np.log(2/args.l_delta)/(2*net_l_qtys[i])) for i in range(args.l_devices)]
sqrt_t = [3*np.sqrt(np.log(2/args.l_delta)/(2*all_u_qtys[i])) for i in range(args.u_devices)]

sqrt_alld = sqrt_s+sqrt_t

# %% objective fxn - term 1 and 2 [source and target errors]
## fxn name --> posy_err_calc
def err_calc(psi,chi,chi_init,psi_init,err_type,alpha_init=None,div_flag=False,\
               rads=rad_alld,sqrts=sqrt_alld,hat_ep=hat_ep_alld,\
                ep_mis=ep_mismatch,args=args):
    err_denoms = []
    if err_type == 's':
        chi_scale = np.array(hat_ep) + 2*np.array(rads) + np.array(sqrts)
        
        chi_init = np.divide(chi_init,chi_scale)
        chi_var_scale = [chi[itsc]/tcs for itsc,tcs in enumerate(chi_scale)] #np.divide(chi,chi_scale)
        ovr_init = psi_init + chi_init
        
        for i in range(args.t_devices):
            err_denom = cp.power(psi[i]*ovr_init[i]/psi_init[i],\
                                   psi_init[i]/ovr_init[i])
            err_denom *= cp.power(chi_var_scale[i]*ovr_init[i]/chi_init[i], \
                                    chi_init[i]/ovr_init[i])
            err_denoms.append(err_denom)
            
        return err_denoms,chi_scale #chi_scale for debug
    elif err_type == 't':
        chi_scale = {} # need a unique scaling for each i,j pair
        chi_scale_init = {} # for alpha_init
        err_denoms = {}
        # chi_init_lists = []
        for j in range(args.t_devices):
            chi_scale[j] = []
            chi_scale_init[j] = []
            for i in range(args.t_devices):
                if div_flag == False:
                    t_chi_scale = alpha[i,j]*(hat_ep[i]+2*rad_alld[i]+sqrts[i]\
                                +ep_mis[j][i]+4*rad_alld[j]+sqrts[j])
                    t_chi_scale_init = alpha_init[i,j]*(hat_ep[i]+\
                                2*rad_alld[i]+sqrts[i]\
                                +ep_mis[j][i]+4*rad_alld[j]+sqrts[j])
                else:
                    raise TypeError('not in yet')

                chi_scale[j].append(t_chi_scale)     
                chi_scale_init[j].append(t_chi_scale_init)
            chi_scale[j] = np.array(chi_scale[j])
            
            temp_chi_init = np.divide(chi_init[:][j],chi_scale[j])
            # chi_init_lists.append(temp_chi_init)
            chi[:,j] = cp.multiply(chi[:,j],cp.power(cp.hstack(chi_scale[j]),-1))
            ovr_init = psi_init + temp_chi_init #chi_init[:,j]
            
            err_denoms[j] = []
            for i in range(args.t_devices):
                err_denom = cp.power(psi[i]*ovr_init[i]/psi_init[i],\
                                     psi_init[i]/ovr_init[i])
                err_denom *= cp.power(chi[i,j]*ovr_init[i]/chi_init[i,j],\
                                     temp_chi_init[i,j]/ovr_init[i])
                err_denoms[j].append(err_denom)
                
            return err_denoms,chi_scale,chi_scale_init
    else:
        raise TypeError('Wrong error type')
    
## auxiliary variables
chi_s = cp.Variable(args.t_devices,pos=True) 
chi_t = cp.Variable((args.t_devices,args.t_devices),pos=True)

# %% objective fxn - term 3 [energy costs]
e_err = 1e-6


# %% constraints
constraints = []

# alpha constraints
con_alpha = []
for i in range(args.t_devices):
    for j in range(args.t_devices):
        con_alpha.append(alpha[i,j] <= 1)
    con_alpha.append(cp.sum(alpha[:,i]) <= 1)
constraints.extend(con_alpha)

# psi constraints
con_psi = []
for i in range(args.t_devices):
    con_psi.append(psi[i] <= 1)
constraints.extend(con_psi)

# fxn to repeat build source constraints
def build_posy_cons(denoms,args=args):
    s_list = []
    
    for c_denom in denoms:
        s_list.append(1/c_denom <= 1)
    return s_list


# %% posynomial approximation fxn
def posy_init(iter_num,cp_vars,args=args):
    ## iter_num - iteration number
    ## cp_vars - dict of optimization variables on which to perform posynomial approximation
    if iter_num == 0:
        psi_init = 0.5*np.ones(args.t_devices)
        chi_s_init = 10*np.ones(args.t_devices)
        chi_t_init = (10*np.ones((args.t_devices,args.t_devices))).tolist()
        alpha_init = 1e-2*np.ones((args.t_devices,args.t_devices))
    else:
        psi_init = cp_vars['psi'].value
        chi_s_init = cp_vars['chi_s'].value
        chi_t_init = cp_vars['chi_t'].value
        alpha_init = cp_vars['alpha'].value

    return psi_init,chi_s_init,chi_t_init,alpha_init

# %% combine and run
for c_iter in range(args.approx_iters):
    t_dict = {'psi':psi,'chi_s':chi_s,'chi_t':chi_t,'alpha':alpha}
    psi_init,chi_s_init,chi_t_init,alpha_init = posy_init(c_iter,t_dict)
    
    # source error term
    s_denoms,chi_s_scale = err_calc(psi,chi_s,chi_s_init,psi_init,err_type='s')
    s_posy_con = build_posy_cons(s_denoms)
    s_err = phi_s*cp.sum(chi_s)
    
    # target error term
    t_denoms,chi_t_scale,chi_t_scale_init = err_calc(psi,chi_t,chi_t_init,psi_init,err_type='t',\
                        alpha_init=alpha_init)
    t_posy_con = build_posy_cons(t_denoms)
    t_err = phi_t*cp.sum(chi_t)
    
    posy_con = s_posy_con + t_posy_con
    net_con = constraints+posy_con
    # net_con = constraints+s_posy_con
    
    obj_fxn = 1e-6+s_err#+t_err
    prob = cp.Problem(cp.Minimize(obj_fxn),constraints=net_con)
    prob.solve(solver=cp.MOSEK,gp=True)#,verbose=True)
    
    # print checking
    print(prob.value)














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


## s_err_calc adjustment for c_iter = 0
    # if c_iter == 0: # (1-psi[i])*(hat_ep[i]+2*rads[i]+sqrts[i])    
    #     ovr_init = psi_init + chi_init
    # else: #c_iter > 0
    #     psi_init = psi.value
    #     chi_init = np.divide(chi.value,chi_scale)


# # %% objective fxn - term 2 [target error] 

# t_err = 1e-6
# for j in range(args.t_devices):
#     temp_calc = 1e-6
#     for i in range(args.t_devices):
#         if args.div_flag == False:
#             temp_calc += (1-psi[i])*alpha[i,j]*\
#                 (hat_ep_alld[i]+2*rad_alld[i]+sqrt_alld[i]+\
#                 4*rad_alld[i]+4*rad_alld[j]+6*sqrt_alld[i]+6*sqrt_alld[j])
#         else:
#             raise TypeError('no work atm')
        
#     t_err += psi[j]*temp_calc


