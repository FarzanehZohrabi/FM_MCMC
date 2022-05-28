#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18  12:18:14 2022

@author: farzanehzohrabi
"""
#This is a code for loading the Fisher Matrix and creating the Covariences

import numpy as np
from numpy.linalg import eig
import emcee
import MulensModel as mm
import random
import matplotlib.pyplot as plt
import pandas as pd
import corner
from PyAstronomy import pyasl
import functions



with open("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82_%s.det.fm.0" %case,"r") as fm:
    data = fm.read()
data = data.splitlines() # importing the fm.0 file to be read

df = pd.DataFrame({'name':data}) # converting list into data frame
df = df.name.str.split(expand=True)
df_bij = df[1:7]
bij = df_bij.to_numpy(dtype=float)
#bij = bij[:, np.newaxis]
df = df[25::] # last three rows with the important parameters 
params_fm = df.to_numpy()

# defining parameter and uncertainty values
t_0 = float(params_fm[1,0])
t_E = float(params_fm[1,1])
u_0 = float(params_fm[1,2])
F0 = float(params_fm[1,4])
fs = float(params_fm[1,5])
sigma_t0 = float(params_fm[2,0])
sigma_tE = float(params_fm[2,1])
sigma_u0 = float(params_fm[2,2])
sigma_F0 = float(params_fm[2,4])
sigma_fs = float(params_fm[2,5])

params = dict() # creating a dictionary of linear parameters
params['t_0'] = float(params_fm[1,0])
params['t_E'] = float(params_fm[1,1])
params['u_0'] = abs(float(params_fm[1,2]))
params['F0'] = float(params_fm[1,4])
params['fs'] = float(params_fm[1,5])

params2 = dict() # creating a dictionary of linear/logarithmic parameters
params2['t_0'] = t_0
params2['logt_E'] = np.log10(t_E)
params2['logu_0'] = np.log10(abs(u_0))
params2['F0'] = F0
params2['log_fs'] = np.log10(fs)

mm_params = dict()
mm_params['t_0'] = float(params_fm[1,0])
mm_params['t_E'] = float(params_fm[1,1])
mm_params['u_0'] = abs(float(params_fm[1,2]))
mm_params['pi_E_N'] = 0.
mm_params['pi_E_E'] = 0.

parameters_to_fit = ["t_0", "t_E", "u_0", "F0", "fs"]
print (parameters_to_fit)
sigmas = [float(params_fm[2,0]), float(params_fm[2,1]), float(params_fm[2,2]),
              float(params_fm[2,4]), float(params_fm[2,5])]
print(sigmas)        
bij_cols = np.column_stack((bij[0:,:3], bij[0:,4:]))
x = bij_cols[0:3,:]
y = bij_cols[4:,:]
bij = np.row_stack((x,y)) # bij with rs removed   (why?)

if u_0 < 0:
    bij[:,2] = -bij[:,2]
    bij[2,:] = -bij[2,:]
    u_0 = abs(u_0)

if u_0 < sigma_u0:
    print('log case')
    # Which parameters we want to fit?
    parameters_to_fit = ["t_0", "logt_E", "logu_0", "F0", "log_fs"] 
                 
    # Initializations for EMCEE
    n_dim = len(parameters_to_fit)
    n_walkers = 40
    n_steps = 3000
    n_burn = 500
    # Including the set of n_walkers starting points:
    start_1 = [params2[p] for p in parameters_to_fit]
    start = [start_1 + np.random.randn(n_dim) * sigmas
              for i in range(n_walkers)]
    # Sigmas for each variable 
    logsig_tE = sigma_tE/(np.log(10)*t_E)
    logsig_u0 = sigma_u0/(np.log(10)*u_0)
    logsig_fs = sigma_fs/(np.log(10*fs))

    if logsig_u0 > 1:
        logsig_u0 = 0.3
        
    if logsig_tE > 1:
        logsig_tE = 0.3
        
    if logsig_fs > 1:
        logsig_fs = 0.3

    sigmas = [sigma_t0, logsig_tE, logsig_u0, sigma_F0, logsig_fs]
    cij_inv = np.linalg.inv(bij)
    print (parameters_to_fit)
    print(sigmas)


if u_0 > sigma_u0:
    print('linear case')
    
    # Which parameters we want to fit?
    parameters_to_fit = ["t_0", "t_E", "u_0", "F0", "fs"]
    
    sigmas = [float(params_fm[2,0]), float(params_fm[2,1]), float(params_fm[2,2]),
              float(params_fm[2,4]), float(params_fm[2,5])]
    cij_inv = np.linalg.inv(bij)
    print (parameters_to_fit)
    print(sigmas)

# if u_0 > sigma_u0:
#     print('linear case')
    
#     # Which parameters we want to fit?
#     parameters_to_fit = ["t_0", "t_E", "u_0", "F0", "fs"]
    
#     sigmas = [float(params_fm[2,0]), float(params_fm[2,1]), float(params_fm[2,2]),
#               float(params_fm[2,4]), float(params_fm[2,5])]
    
#     # Initializations for EMCEE
#     n_dim = len(parameters_to_fit)
#     n_walkers = 40
#     n_steps = 3000
#     n_burn = 500
#     # Including the set of n_walkers starting points:
#     start_1 = [params[p] for p in parameters_to_fit]
#     start = [start_1 + np.random.randn(n_dim) * sigmas
#               for i in range(n_walkers)]
    
#     # Run EMCEE (this can take some time):
#     sampler = emcee.EnsembleSampler(
#         n_walkers, n_dim, ln_prob, args=(lc_data[:,0], lc_data[:,1], lc_data[:,2]))
#     sampler.run_mcmc(start, n_steps)
    
#     samples0 = sampler.chain[:, :, :].reshape((-1, n_dim))
#     samples_data = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
#     samples = samples_data
#     samples[:,2] = abs(samples[:,2])
    
#     cij_inv1 = np.linalg.inv(bij)

# if (areSame(cij_inv, cij_inv1)==1):
#    print("Matrices are identical")
# else:
#    print("Matrices are not identical")






# My Model
#my_model = mm.Model(mm_params, coords=coords)


#     'q': q, 's': rs, 'alpha':alpha})
#PS_model.plot_magnification(
#    t_range=[t_1, t_2], subtract_2450000=False, color='red', 
#    linestyle=':', label='PSPL')



#tr_time_mag = np.transpose(time_mag)
#MM_mag_data = mm.MulensData(tr_time_mag)
#my_model = mm.Model(mm_params ,coords=coords)
#my_event = mm.Event(datasets=MM_mag_data, model= PS_model)






#  # generate fitted models
# rand_samps = []
# for i in range(10):
#     rand_samps.append(random.randrange(1, 14001, 1))
# models = []
# plt.figure()
# plt.title('Data and Fitted Models (%s)' %case)
# plot_params = {'lw': 2.5, 'alpha': 0.3, 'subtract_2450000': False,
#                 't_start': t_1, 't_stop': t_2}
# # t_1 and t_2 act as the limits on the x-axis to centralize the peak
# my_event.plot_data(subtract_2450000=False)
# plt.xlim(t_1,t_2)
# for i in range(10):
#     models = mm.Model({'t_0': samples[rand_samps[i],0], 't_E': abs(samples[rand_samps[i],1]),
#               'u_0': abs(samples[rand_samps[i],2])}, coords=coords)
#     models.set_datasets([MM_mag_data])
#     models.plot_lc(**plot_params)
# plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/lc_plots/SL_%s_models.jpg" %case)