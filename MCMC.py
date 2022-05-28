#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 07:46:07 2022

@author: farzanehzohrabi
"""

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


if u_0 < sigma_u0:
    print('log case')
    # Which parameters we want to fit?
    parameters_to_fit = ["t_0", "logt_E", "logu_0", "F0", "log_fs"]
    parameters_to_fit1 = [t_0, np.log10(t_E), np.log10(abs(u_0)), F0,np.log10(fs)] 
    print(parameters_to_fit1)           
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
    print(sigmas)
    # Initializations for EMCEE
    n_dim = len(parameters_to_fit)
    n_walkers = 40
    n_steps = 3000
    n_burn = 500
    # Including the set of n_walkers starting points:
    start_1 = [params2[p] for p in parameters_to_fit]
    start = [start_1 + np.random.randn(n_dim) * sigmas
              for i in range(n_walkers)]
    
      # Run EMCEE (this can take some time):
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_prob_log, args=(lc_data[:,0], lc_data[:,1], lc_data[:,2]))
    sampler.run_mcmc(start, n_steps)
    
    samples0 = sampler.chain[:, :, :].reshape((-1, n_dim))
    samples_data = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
    samples = samples_data
    samples[:,1] = 10**(samples[:,1])
    samples[:,2] = 10**(samples[:,2])
    samples[:,4] = 10**(samples[:,4])
    print(samples[1,1],samples[1,2],samples[1,4] )
if u_0 > sigma_u0:
    print('linear case')
    
    # Which parameters we want to fit?
    parameters_to_fit = ["t_0", "t_E", "u_0", "F0", "fs"]
    print(parameters_to_fit)
    sigmas = [float(params_fm[2,0]), float(params_fm[2,1]), float(params_fm[2,2]),
              float(params_fm[2,4]), float(params_fm[2,5])]
    print(sigmas)
    # Initializations for EMCEE
    n_dim = len(parameters_to_fit)
    n_walkers = 40
    n_steps = 3000
    n_burn = 500
    # Including the set of n_walkers starting points:
    start_1 = [params[p] for p in parameters_to_fit]
    start = [start_1 + np.random.randn(n_dim) * sigmas
              for i in range(n_walkers)]
    
    # Run EMCEE (this can take some time):
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_prob, args=(lc_data[:,0], lc_data[:,1], lc_data[:,2]))
    sampler.run_mcmc(start, n_steps)
    
    samples0 = sampler.chain[:, :, :].reshape((-1, n_dim))
    samples_data = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
    samples = samples_data
    samples[:,2] = abs(samples[:,2])
    print(samples[1,2])
chains_tr = np.transpose(samples)
cij_chains= np.cov(chains_tr) 

# plotting covariance matrices
cov_ellipses_ch = np.zeros((1,50))
image_no = 1
for i in range(4):
    for j in range(5):
        if i != j:
            cov = np.array([[cij_chains[i,i], cij_chains[i,j]], [cij_chains[j,i], 
                                                                 cij_chains[j,j]]])
            vals,vects = eig(cov)
            maxcol = list(vals).index(max(vals))
            mincol = list(vals).index(min(vals))
            max_eigenvect = vects[:,maxcol]
            min_eigenvect = vects[:,mincol]
            
            alpha = np.arctan2(max_eigenvect[1], max_eigenvect[0])

            if alpha < 0:
                alpha = alpha + 2*np.pi
            
            med = np.median(samples,axis=0)
            X = med[i]
            Y = med[j]
            
            a = maj_axes(vals) # length of major axis
            b = min_axes(vals) # length of minor axis
            
            R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            
            ellipse_x_r = a*np.cos(theta_grid)
            ellipse_y_r = b*np.sin(theta_grid)
            
            ellipse = np.concatenate(([ellipse_x_r],[ellipse_y_r]))
            r_ellipse = R.dot(ellipse)
            cov_all_ch = np.concatenate((cov_ellipses_ch, r_ellipse), axis=0)
            cov_ellipses_ch = cov_all_ch
            
            # plt.figure()
            # plt.plot(samples[:,i], samples[:,j],'.',color='red', ms=3, alpha=0.1)
            # plt.plot(r_ellipse[0,:] + X, r_ellipse[1,:] + Y)
            # plt.title('Covariance Error Ellipse(chains) and Data (SL %s)' %case)
            # plt.xlabel(parameters_lin[i])
            # plt.ylabel(parameters_lin[j])
            # plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/SL_MCMC_Error_Ellipses/SL_%s_cov_ch_%s.jpg" %(case, str(image_no)))
            image_no = image_no + 1

cov_all_ch = cov_all_ch[1:,:]
cov_all_ch = np.delete(cov_all_ch, (8,9,16,17,18,19,24,25,26,27,28,29), axis=0)






cov_ellipses_inv = np.zeros((1,50)) 
for i in range(4):
    for j in range(5):
        if i != j:
            cov = np.array([[cij_inv[i,i], cij_inv[i,j]], [cij_inv[j,i], 
                                                                 cij_inv[j,j]]])
            vals,vects = eig(cov)
            maxcol = list(vals).index(max(vals))
            mincol = list(vals).index(min(vals))
            max_eigenvect = vects[:,maxcol]
            min_eigenvect = vects[:,mincol]
            
            alpha = np.arctan2(max_eigenvect[1], max_eigenvect[0])

            if alpha < 0:
                alpha = alpha + 2*np.pi
            med = np.median(samples,axis=0)
            X = med[i]
            Y = med[j]
            
            a = maj_axes(vals) # length of major axis
            b = min_axes(vals) # length of minor axis
            
            R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            
            ellipse_x_r = a*np.cos(theta_grid)
            ellipse_y_r = b*np.sin(theta_grid)
            
            ellipse = np.concatenate(([ellipse_x_r],[ellipse_y_r]))
            r_ellipse = R.dot(ellipse)
            cov_all_inv = np.concatenate((cov_ellipses_inv, r_ellipse), axis=0)
            cov_ellipses_inv = cov_all_inv
            
            #plt.figure()
            #plt.plot(samples[:,i], samples[:,j],'.',color='red', ms=3, alpha=0.1)
            #plt.plot(r_ellipse[0,:] + X, r_ellipse[1,:] + Y)
            #plt.title('Covariance Error Ellipse(inv) and Data (SL %s)' %case)
            #plt.xlabel(parameters_lin[i])
            #plt.ylabel(parameters_lin[j])
            #plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/SL_FM_Error_Ellipses/SL_%s_cov_inv_%s.jpg" %(case, str(image_no)))
            image_no = image_no + 1
           
                
cov_all_inv = cov_all_inv[1:,:]
cov_all_inv = np.delete(cov_all_inv, (8,9,16,17,18,19,24,25,26,27,28,29), axis=0)

corner_plt_inv()
#inv_MCMC_plt()
#Call the function that creates corner plots
corner_plt_MCMC()

#ratio of cij_inv/cij_chains
cij_ratio = cij_inv/cij_chains

# # Results:
results = np.percentile(samples, [15.8, 50, 84.1], axis=0)
# axis=0 goes down the columns
print("Fitted parameters:")
for i in range(n_dim):
    r = results[1, i]
    print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]))


# # directing the data to a txt file using a txt_case() function
#txt_case()
#stat()