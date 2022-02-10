
main_script.py backups


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


## non-posynomial two mismatch prevention constraints
# for j in range(args.t_devices):
#     # con_prev.append(cp.sum(alpha[:,j]) <= 1e-3+psi[j]) # this needs another posynomial approximation
#     for i in range(args.t_devices):
#     #     con_prev.append((1+psi[i]-psi[j])*alpha[i,j] <= 1e-3)
# constraints.extend(con_prev)


## old posy_denom calc
    # t_con_denoms = []
    # ovr_init = np.array(chi_init) +  np.array(psi_init)
    # for j in range(args.t_devices):
    #     t1_con_denoms = cp.power(chi[j]*ovr_init[j]/chi_init[j], \
    #                              chi_init[j]/ovr_init[j])
    #     t2_con_denoms = cp.power(psi[j]*ovr_init[j]/psi_init[j], \
    #                              psi_init[j]/ovr_init[j])
    #     t_con_denoms.append(t1_con_denoms*t2_con_denoms)


## old build_posy_ts_con
    # t_con_prev = []
    # if cp_type == 1:
    #     for j in range(args.t_devices):
    #         for i in range(args.t_devices):
    #             t_num = alpha[i,j]*(1+psi[i])
    #             t_con_prev.append(t_num/denoms[j] <= 1) 
    # elif cp_type == 2:
    #     for j in range(args.t_devices):
    #         t_con_prev.append(cp.sum(alpha[:,j])/denoms[j] <= 1)
    # else:
    #     raise TypeError('cptype invalid posy con')

# posynomial approx for constraints
            # pos_t1_denom = cp.power(chi*ovr_init_pos[j]/chi_init, \
            #                      chi_init/ovr_init_pos[j])
            # pos_t2_denom = cp.power(psi[j]*ovr_init_pos[j]/psi_init[j], \
            #                      psi_init[j]/ovr_init_pos[j])
            # pos_t3_denom = cp.power(ovr_init_pos[j], \
            #                      cp_epsilon/ovr_init_pos[j])
            # pos_t_con_denoms.append(pos_t1_denom*pos_t2_denom*pos_t3_denom)

        # ovr_init_pos = np.array(chi_init) + np.array(psi_init) \
        #     + cp_epsilon*np.ones(args.t_devices)

# %% 
# if div_flag == False:
#     t_chi_scale = alpha[i,j]*(hat_ep[i]+2*rad_alld[i]+sqrts[i]\
#                 +ep_mis[j][i]+4*rad_alld[j]+sqrts[j])
#     t_chi_scale_init = alpha_init[i,j]*(hat_ep[i]+\
#                 2*rad_alld[i]+sqrts[i]\
#                 +ep_mis[j][i]+4*rad_alld[j]+sqrts[j])
# else:
#     if i == j:
#         t_chi_scale = alpha[i,j]*(hat_ep[i]+2*rad_alld[i]+sqrts[i]\
#                     +ep_mis[j][i]+4*rad_alld[j]+sqrts[j]+\
#                     0.5*1+\
#                     2*(rad_alld[i]+rad_alld[j]) + \
#                     sqrts[i]+sqrts[j])
#         t_chi_scale_init = alpha_init[i,j]*(hat_ep[i]+\
#                     2*rad_alld[i]+sqrts[i]\
#                     +ep_mis[j][i]+4*rad_alld[j]+sqrts[j]+\
#                     0.5*1+\
#                     2*(rad_alld[i]+rad_alld[j]) + \
#                     sqrts[i]+sqrts[j])                
#     else:
#         t_chi_scale = alpha[i,j]*(hat_ep[i]+2*rad_alld[i]+sqrts[i]\
#                     +ep_mis[j][i]+4*rad_alld[j]+sqrts[j]+\
#                     0.5*div_vals[i,j]/100+\
#                     2*(rad_alld[i]+rad_alld[j]) + \
#                     sqrts[i]+sqrts[j])
#         t_chi_scale_init = alpha_init[i,j]*(hat_ep[i]+\
#                     2*rad_alld[i]+sqrts[i]\
#                     +ep_mis[j][i]+4*rad_alld[j]+sqrts[j]+\
#                     0.5*div_vals[i,j]/100+\
#                     2*(rad_alld[i]+rad_alld[j]) + \
#                     sqrts[i]+sqrts[j])










