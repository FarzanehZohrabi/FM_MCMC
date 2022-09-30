#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20  12:18:14 2022

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
import re
import glob
import os.path
import datetime
from matplotlib.backends.backend_pdf import PdfPages

"Functions"


def case_stats():
    with open("/Users/farzanehzohrabi/Documents/MCMC/test1/SL_stats.txt" , "a+") as stat:
        result_MCMC= []
        for i in range(n_dim):
            r = results[1, i]
            result = [r, results[2, i]-r, r-results[0, i]]
            result_MCMC.append(result)
        result_MCMC = np.array(result_MCMC)
        t_E_MC = result_MCMC[1,0]

        if result_MCMC[1,1] > result_MCMC[1,2]:
            sig_tE_MC = result_MCMC[1,1]
        else:
            sig_tE_MC = result_MCMC[1,2]
        
        sigma_cal = [[i] for i in sig_calculated]
        
         
        if Linear_case==False:
            stat.write("{} single_lens logarithmic_case ".format(case[z]))
            print("single_lens logarithmic_case ")
            if (sig_calculated[1]) <= (1/3) and (sig_tE_MC)<= (1/3):
                stat.write("well-constrained agree_well "+ "| ")
                print("well-constrained agree_well ")
            elif (sig_calculated[1]) <= (1/3) and (sig_tE_MC)> (1/3):
                stat.write("well-constrained disagree_fmw_MCp "+ "| ")
                print("well-constrained disagree_fmw_MCp")
            elif (sig_calculated[1]) > (1/3) and (sig_tE_MC) > (1/3):
                stat.write("poorly-constrained agree_poor "+ "| ")
                print("poorly-constrained agree_poor ")
            elif (sig_calculated[1]) > (1/3) and (sig_tE_MC) <= (1/3):
                stat.write("poorly-constrained disagree_fmp_MCw "+ "| ") 
                print("poorly-constrained disagree_fmp_MCw ")
        
        if Linear_case==True:
            stat.write("{} single_lens linear_case ".format(case[z]))
            print("single_lens linear_case")
            if (sig_calculated[1]/t_E) <= (1/3) and (sig_tE_MC/t_E_MC)<= (1/3):
                stat.write("well-constrained agree_well "+ "| ")
                print("well-constrained agree_well")
            elif (sig_calculated[1]/t_E) <= (1/3) and (sig_tE_MC/t_E_MC) > (1/3):
                stat.write("well-constrained disagree_fmw_MCp "+ "| ")
                print("well-constrained disagree_fmw_MCp ")
            elif (sig_calculated[1]/t_E) > (1/3) and (sig_tE_MC/t_E_MC) > (1/3) :
                stat.write("poorly-constrained agree_poor "+ "| ")
                print("poorly-constrained agree_poor")
            elif (sig_calculated[1]/t_E) > (1/3) and (sig_tE_MC/t_E_MC) <= (1/3) :
                stat.write("poorly-constrained disagree_fmp_MCw "+ "| ")
                print("poorly-constrained disagree_fmp_MCw ")
        
        for row in np.round(fm_params, decimals = 6):
            stat.write(" ".join(str(item) for item in row)+ " ")
        stat.write("| ")
        for l in np.round(sigma_cal, decimals = 6):
            stat.write(" ".join(str(item) for item in l)+ " ")
        stat.write("| ")
        for j in np.round(result_MCMC, decimals = 6):
            stat.write(" ".join(str(item) for item in j)+ " ")
        
        stat.write("\n")

def case_info():
    with open("/Users/farzanehzohrabi/Documents/MCMC/test1/SL_%s.txt" %case[z], "a") as f:
        if Linear_case==True:
            print("linear case", file=f)
        if Linear_case==False:
            print("log case", file=f)
        if (sig_calculated[1]) <= (1/3):
            print("well-constrained case", file=f)
        if (sig_calculated[1]) > (1/3):
            print("poorly-constrained case", file=f)
        print("bij:", bij, file=f) 
        print("sigma calculated:", sig_calculated, file=f)
        print("cij (from inverting bij):", cij_inv, file=f)
        print("cij (from chains):", cij_chains, file=f)
        print("ratio of cij's (cij_inv/cij_chains):", cij_ratio, file=f)
        print("Fitted parameters:", file=f)
        for i in range(n_dim):
            r = results[1, i]
            print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]), file=f)


def ln_like(theta, t, f, f_err):
    t_0, t_E, u_0, F0, fs = theta
    u = (u_0**2 + ((t-t_0)/t_E)**2)**0.5
    A_t = (u**2 + 2)/(u*(u**2+4)**0.5)
    Fs = fs*F0; Fb = F0*(1-fs)
    F_t = Fs*A_t + Fb
    model = F_t
    sigma2 = f_err ** 2
    return -0.5 * np.sum((f - model)**2/sigma2)
def ln_like_log(theta, t, f, f_err):
    t_0, log_tE, log_u0, F0, log_fs = theta
    t_E = 10**(log_tE)
    u_0 = 10**(log_u0)
    fs = 10**(log_fs)
    u = (u_0**2 + ((t-t_0)/t_E)**2)**0.5
    A_t = (u**2 + 2)/(u*(u**2+4)**0.5)
    Fs = fs*F0; Fb = F0*(1-fs)
    F_t = Fs*A_t + Fb
    model = F_t
    sigma2 = f_err ** 2
    return -0.5 * np.sum((f - model)**2/sigma2)
def ln_prior_log(theta):
    t_0, log_tE, log_u0, F0, log_fs = theta
    if -5 < log_fs < 0.1:
        return -1.0e15
    if -2.0 < log_tE < 4.0:
        return -1.0e15
    if -4.0 < log_u0 < 1.0:
        return -1.0e15
    return 0.0
def ln_prior_lin(theta):
    t_0, t_E, u_0, F0, fs = theta
    return 0.0
def ln_prob_lin(theta, t, f, f_err):
    lp = ln_prior_lin(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnprobval = lp + ln_like(theta, t, f, f_err)
    return lnprobval
def ln_prob_log(theta, t, f, f_err):
    lp = ln_prior_log(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnprobval = lp + ln_like_log(theta, t, f, f_err)
    if not np.isfinite(lnprobval):
        return -np.inf
    return lnprobval
def func(theta, t, f, f_err):
    try:
        lnprobval = -2. * ln_prob_log(theta, t, f, f_err)
    except ValueError: # NaN value case
        lnprobval = -np.inf # just set to negative infinity 
    return lnprobval
def uncertainties(x):
    """
    change the parameters so that we can easily print median and uncertainties
    (based on percentiles)
    """
    return (x[1], x[2]-x[1], x[1]-x[0])
def lin_model(t, theta):
    t_0, tE, u0, F0, fs = theta
    u = (u_0**2 + ((t-t_0)/t_E)**2)**0.5
    A_t = (u**2 + 2)/(u*np.sqrt(u**2+4))
    Fs = fs*F0
    Fb = F0*(1-fs) 
    #F_t = np.nansum(Fs*A_t,Fb)
    F_t =  Fs*A_t + Fb
    no_mm_model = F_t
    return no_mm_model
def log_model(t, theta):
    t_0, log_tE, log_u0, F0, log_fs = theta
    u_0 = 10**(log_u0)
    t_E = 10**(log_tE)
    fs = 10**(log_fs)
    u = (u_0**2 + ((t-t_0)/t_E)**2)**0.5
    A_t = (u**2 + 2)/(u*np.sqrt(u**2+4))
    Fs = fs*F0
    Fb = F0*(1-fs)
    #F_t = np.sum(Fs*A_t,Fb)
    F_t = Fs*A_t + Fb
    no_mm_model = F_t
    return no_mm_model
def MM_lin_model(t, theta):
    t_0, tE, u0, F0, fs = theta
    PS_model = mm.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E})
    fs = start_1[-1]
    F0 = start_1[-2]
    Fs = fs*F0
    Fb = F0*(1-fs)
    #F_t = np.sum(Fs*PS_model.magnification(t),Fb)
    F_t = Fs*PS_model.magnification(t) + Fb
    return F_t
def MM_log_model(t, theta):
    t_0, log_tE, log_u0, F0, log_fs = theta
    u_0 = 10**(log_u0)
    t_E = 10**(log_tE)
    fs = 10**(log_fs)
    PS_model = mm.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E})
    fs = start_1[-1]
    F0 = start_1[-2]
    Fs = fs*F0
    Fb = F0*(1-fs)
    #F_t = np.nansum(Fs*PS_model.magnification(t),Fb)
    F_t = Fs*PS_model.magnification(t) + Fb
    return F_t

"input"
directory = "/Users/farzanehzohrabi/Documents/MCMC/singleLens/"
lcfile = sorted(glob.glob(directory + '/*.det.lc'))
fmfile = sorted(glob.glob(directory + '/*.det.fm.0'))
od = pd.DataFrame(np.loadtxt("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82.out", dtype=str))
deltasqr = od.iloc[:,[0,1,2,3,4,5,6,7,76]] #make a variable here
case= od[0]
delta_sqr= od[76].astype(float)
out_data = od.to_numpy()

#"""Reading LC, FM, and Out file"""
# and lcfile
for z in range(len(case)):
    #print(z)
    if z!= 662:
          continue
    if not delta_sqr[z] < 500.0 and not os.path.exists(directory + "singleLens_0_82_%s.det.fm.0"%case[z])== False:
        with open("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82_%s.det.fm.0" %case[z],"r") as fm:
            data = fm.read()
        data = data.splitlines() # importing the fm.0 file to be read

        df = pd.DataFrame({'name':data}) # converting list into data frame
        df = df.name.str.split(expand=True)
        df_bij = df[1:7]
        bij = df_bij.to_numpy(dtype=float)
        df = df[25::] # last three rows with the important parameters
        params_fm = df.to_numpy()
        
        with open("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82_%s.det.lc" %case[z],"r") as lc:
            lc_data = lc.read()
        lc_data = lc_data.splitlines()
        lc = pd.DataFrame({'name':lc_data}) # converting list into data frame
        lc = lc.name.str.split(expand=True)
        lc_file = lc.to_numpy()
        lc_data = lc[8::]
        lc_data = lc_data.to_numpy(dtype=float)
        lc_data = lc_data[:,0:3]
        
                # defining parameter and uncertainty values
        t_0 = float(params_fm[1,0])
        t_E = float(params_fm[1,1])
        u_0 = float(params_fm[1,2])
        rho = float(params_fm[1,3])
        F0 = float(params_fm[1,4])
        fs = float(params_fm[1,5])
        log_tE = np.log10(t_E)
        logu_0 = np.log10(abs(u_0))
        log_fs = np.log10(fs)
        sigma_t0_fm = float(params_fm[2,0])
        sigma_tE_fm = float(params_fm[2,1])
        sigma_u0_fm = float(params_fm[2,2])
        sigma_rho_fm = float(params_fm[2,3])
        sigma_F0_fm = float(params_fm[2,4])
        sigma_fs_fm = float(params_fm[2,5])
        
        params_linfm = dict() # creating a dictionary of linear parameters
        params_linfm['t_0'] = float(params_fm[1,0])
        params_linfm['t_E'] = float(params_fm[1,1])
        params_linfm['u_0'] = abs(float(params_fm[1,2]))
        params_linfm['rho'] = float(params_fm[1,3])
        params_linfm['F0'] = float(params_fm[1,4])
        params_linfm['fs'] = float(params_fm[1,5])
        
        params_logfm = dict() # creating a dictionary of linear/logarithmic parameters
        params_logfm['t_0'] = t_0
        params_logfm['log_tE'] = np.log10(t_E)
        params_logfm['logu_0'] = np.log10(abs(u_0))
        params_logfm['F0'] = F0
        params_logfm['log_fs'] = np.log10(fs)
        
        parameters_to_fit_lin = ["t_0", "t_E", "u_0", "F0", "fs"]
        parameters_to_fit_log = ["t_0", "log_tE", "logu_0", "F0", "log_fs"]
        params_lin= np.array([[t_0],[t_E],[abs(u_0)],[ F0],[ fs]])
        params_log = np.array([[t_0],[np.log10(t_E)],[np.log10(abs(u_0))],[F0],[np.log10(fs)]])
        # params_linear= np.array([t_0,t_E,abs(u_0), F0, fs])
        # params_logarithmic = np.array([t_0,np.log10(t_E),np.log10(abs(u_0)),F0,np.log10(fs)])
        if u_0 < 0:
            bij[:,2] = -bij[:,2]
            bij[2,:] = -bij[2,:]
            u_0 = abs(u_0)
        cij_inv_w_rho = np.linalg.inv(bij)
        sig_cij_w_rho = np.array(np.sqrt(np.diag(cij_inv_w_rho)),dtype = object)

        bij_cols = np.column_stack((bij[0:,:3], bij[0:,4:]))
        x = bij_cols[0:3,:]
        y = bij_cols[4:,:]
        bij = np.row_stack((x,y)) # bij with rs removed
        
        if not sig_cij_w_rho[3]< 3 * rho:
            cij_inv_lin = np.linalg.inv(bij)
            sig_cij_lin = np.array(np.sqrt(np.diag(cij_inv_lin)),dtype = object)  

        sig_cal = dict() # creating a dictionary of linear parameters
        sig_cal['sigcal_t_0'] = float(sig_cij_lin[0])
        sig_cal['sigcal_t_E'] = float(sig_cij_lin[1])
        sig_cal['sigcal_u_0'] = (float(sig_cij_lin[2]))
        sig_cal['sigcal_F0'] = float(sig_cij_lin[3])
        sig_cal['sigcal_fs'] = float(sig_cij_lin[4])
        
        if sig_cij_lin[2] < (1/3) * abs(u_0):
            Linear_case = True
            print('linear case')
        else: 
            Linear_case = False
            print('log case')
        
        #linear case
        #params = params_linfm
        params= params_linfm
        parameters_to_fit = parameters_to_fit_lin
        cij_inv = cij_inv_lin
        ln_prob = ln_prob_lin
        model = lin_model
        fm_params = params_lin
        note = " t0= %5.2f \n tE= %5.2f \n u0 = %5.2f \n F= %f \n fs = %5.2f \n $\Delta^2$ = %f" %(t_0,t_E,u_0,F0,fs,delta_sqr[z])
        if Linear_case== False:
            #params = params_logfm
            params= params_logfm
            parameters_to_fit = parameters_to_fit_log
            j_trans = np.array([[1,0,0,0,0],
                  [0, 1/(t_E * np.log(10)), 0, 0, 0],
                  [0,0,1/(abs(u_0)* np.log(10)),0,0],
                  [0,0,0,1,0],
                  [0,0,0,0, 1/(fs * np.log(10))]])
            j_trans = j_trans.astype(float)
            log_bij = np.dot(bij,j_trans) 
            transpose_j = j_trans.transpose()
            mul= np.matmul(cij_inv_lin,transpose_j)
            cij_inv_log = np.matmul(j_trans,mul)
            cij_inv = cij_inv_log
            ln_prob = ln_prob_log 
            model = log_model
            fm_params = params_log
            note = " t0= %5.2f \n ltE= %5.2f \n lu0 = %5.2f \n F= %f \n lfs = %5.2f \n $\Delta^2$ = %5.2f" %(t_0,log_tE,logu_0,F0,log_fs,delta_sqr[z])

        sig_calculated = np.array(np.sqrt(np.diag(cij_inv)),dtype = object)
        
        
        fs = float(lc_file[0,1])
        Zs = float(lc_file[1,1])
        
        m0 = Zs + 2.5*np.log10(fs)
        
        # converting relative flux to magnitude
        m = m0 - 2.5*np.log10(lc_data[:,1])  
        # converting relative flux error to magnitude error
        sigma_m = (2.5/np.log(10))*(lc_data[:,2]/lc_data[:,1])
        time = lc_data[:,0].flatten() # flattening the time array to match the dimensions 
        #of m and sigma_m
        
        time_mag = np.column_stack((time, m, sigma_m))
        ra = float(out_data[int(case[z]),5])
        dec = float(out_data[int(case[z]),6])
        sexa = pyasl.coordsDegToSexa(ra, dec)
        coords = sexa
        
        #EMCEE
        #sigmas = [sig_calculated[0],sig_calculated[1],sig_calculated[2],(1.0e-5),sig_calculated[4]]
        #sigmas = [(1.0e-4)*sig_calculated[0], (1.0e-4)*sig_calculated[1],(1.0e-4)*sig_calculated[2], (1.0e-4)*sig_calculated[3], (1.0e-4)*sig_calculated[4]]
        #sigmas = [(1.0e-4)*sig_calculated[0], (1.0e-4)*sig_calculated[1],(1.0e-4)*sig_calculated[2], (1.0e-5), (1.0e-4)*sig_calculated[4]]
        sigmas = [(1.0e-4), (1.0e-4),(1.0e-4), (1.0e-5), (1.0e-4)] # we chose 10**-4 to  let MCMC guess for itself instead of starting from the sigmas from FM
        if Linear_case==False:
            if sig_calculated[1] > 1:
                sigmas[1] = 0.3
            
            if sig_calculated[2] > 1:
                sigmas[2] = 0.3
            
            if sig_calculated[4] > 1:
                sigmas[4] = 0.3
    
        
        # Initializations for EMCEE
        n_dim = len(parameters_to_fit)
        n_walkers = 50
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
        
        #sampler.get_chain(discard=100, thin=5, flat=True)
        samples0 = sampler.chain[:, :, :].reshape((-1, n_dim))
        # samples_data = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
        samples_data = sampler.get_chain(discard=500, flat=True, thin=20).reshape((-1, n_dim))
        samples = samples_data
        
        
        chains_tr = np.transpose(samples)
        cij_chains= np.cov(chains_tr) 
        
        # ratio of cij_inv/cij_chains
        cij_ratio = cij_inv/cij_chains
        
        results = np.percentile(samples, [15.8, 50, 84.1], axis=0)
        def maj_axes(n):
            return np.sqrt(2.28*max(n),dtype=float)
        def min_axes(n):
            return np.sqrt(2.28*min(n),dtype=float)
        
        theta_grid = np.linspace(0,2*np.pi)
        
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
                    # plt.xlabel(parameters_to_fit[i])
                    # plt.ylabel(parameters_to_fit[j])
                    #plt.savefig("/Users/emersongehr/Desktop/SL_graphs/SL_%s_cov_ch_%s.jpg" %(case, str(image_no)))
                    image_no = image_no + 1
        
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
                    
                    # plt.figure()
                    # plt.plot(samples[:,i], samples[:,j],'.',color='red', ms=3, alpha=0.1)
                    # plt.plot(r_ellipse[0,:] + X, r_ellipse[1,:] + Y)
                    # plt.title('Covariance Error Ellipse(inv) and Data (SL %s)' %case)
                    # plt.xlabel(parameters_to_fit[i])
                    # plt.ylabel(parameters_to_fit[j])
                    #plt.savefig("/Users/emersongehr/Desktop/SL_graphs/SL_%s_cov_inv_%s.jpg" %(case, str(image_no)))
                    image_no = image_no + 1

        cov_all_inv = cov_all_inv[1:,:]
        cov_all_ch = cov_all_ch[1:,:]
        
        cov_all_ch = np.delete(cov_all_ch, (8,9,16,17,18,19,24,25,26,27,28,29), axis=0)
        cov_all_inv = np.delete(cov_all_inv, (8,9,16,17,18,19,24,25,26,27,28,29), axis=0)
        
# Geometry of the event: only really matters for parallax
        q = 0 # single lens: M2/M1 = 0
        m1 = 1.0/(1+q) # m1 = M1/Mtot
        a = 0 # dimensionless separation between lenses: single lens causes a = 0
        rs = float(params_fm[1,3]) # source radius 
        alpha = 0 # single lens event
        slope = alpha*(np.pi/180) # changing alpha into radians 
        cosB = np.cos(slope)
        sinB = np.sin(slope)
        xcom = -m1*a
        
        # defining tau at different times 
        tt = (time - t_0)/t_E
        # defining u for different values of tau
        u = (np.sqrt((u_0**2)+(tt**2)))
        # parameterized equations for x and y 
        xsCenter = u*cosB - tt*sinB + xcom
        ysCenter = u*sinB + tt*cosB
        
        t_1 = t_0 - 400
        t_2 = t_0 + 400
        
        useMM = True
        lin_fluxfun = lin_model
        log_fluxfun = log_model
        if useMM==True:
            lin_fluxfun = MM_lin_model
            log_fluxfun = MM_log_model
        
        pdf = PdfPages('/Users/farzanehzohrabi/Documents/MCMC/test1/SL_%s.pdf' %case[z])  
        with pdf:
            plt.figure()
            fig = corner.corner(samples, labels = parameters_to_fit,levels = (0.68, 0.95, 0.99))
            plt.title('MCMC & FM  %s' %case[z])
            axs = np.array(fig.axes)
            axs[5].plot(cov_all_ch[0,:] + med[0], cov_all_ch[1,:] + med[1], color='blue')
            axs[10].plot(cov_all_ch[2,:] + med[0], cov_all_ch[3,:] + med[2], color='blue')
            axs[11].plot(cov_all_ch[8,:] + med[1], cov_all_ch[9,:] + med[2], color='blue')
            axs[15].plot(cov_all_ch[4,:] + med[0], cov_all_ch[5,:] + med[3], color='blue')
            axs[16].plot(cov_all_ch[10,:] + med[1], cov_all_ch[11,:] + med[3], color='blue')
            axs[17].plot(cov_all_ch[14,:] + med[2], cov_all_ch[15,:] + med[3], color='blue')
            axs[20].plot(cov_all_ch[6,:] + med[0], cov_all_ch[7,:] + med[4], color='blue')
            axs[21].plot(cov_all_ch[12,:] + med[1], cov_all_ch[13,:] + med[4], color='blue')
            axs[22].plot(cov_all_ch[16,:] + med[2], cov_all_ch[17,:] + med[4], color='blue')
            axs[23].plot(cov_all_ch[18,:] + med[3], cov_all_ch[19,:] + med[4], color='blue')
            
            
            axs[5].plot(cov_all_inv[0,:] + med[0], cov_all_inv[1,:] + med[1], color='red', linestyle='dashed')
            axs[10].plot(cov_all_inv[2,:] + med[0], cov_all_inv[3,:] + med[2], color='red', linestyle='dashed')
            axs[11].plot(cov_all_inv[8,:] + med[1], cov_all_inv[9,:] + med[2], color='red', linestyle='dashed')
            axs[15].plot(cov_all_inv[4,:] + med[0], cov_all_inv[5,:] + med[3], color='red', linestyle='dashed')
            axs[16].plot(cov_all_inv[10,:] + med[1], cov_all_inv[11,:] + med[3], color='red', linestyle='dashed')
            axs[17].plot(cov_all_inv[14,:] + med[2], cov_all_inv[15,:] + med[3], color='red', linestyle='dashed')
            axs[20].plot(cov_all_inv[6,:] + med[0], cov_all_inv[7,:] + med[4], color='red', linestyle='dashed')
            axs[21].plot(cov_all_inv[12,:] + med[1], cov_all_inv[13,:] + med[4], color='red', linestyle='dashed')
            axs[22].plot(cov_all_inv[16,:] + med[2], cov_all_inv[17,:] + med[4], color='red', linestyle='dashed')
            axs[23].plot(cov_all_inv[18,:] + med[3], cov_all_inv[19,:] + med[4], color='red', linestyle='dashed')
            plt.show()
            # pdf.savefig()  # saves the current figure into a pdf page
            # plt.close()            
            """fitting random seeds MCMC on the light curve"""
            randomMC = random.sample(list(samples), 50)  
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.text(0.95, 0.95, note,
                    verticalalignment='top', horizontalalignment='right',
                    transform=ax.transAxes,color='green', fontsize=10 ,ma='left')
            ax.set(xlim=(t_1,t_2))   
            ax.errorbar(lc_data[:,0], lc_data[:,1], yerr = lc_data[:,2], fmt="o")
            for k in range(0,19):
                f = model(time,randomMC[k])

                ax.set_title('MCMC random models (%s)' %case[z])
                ax.plot(time,f,alpha=0.5)
            plt.show()
            # pdf.savefig()
            # plt.close()
            
            
            """No MM Model"""
            plt.figure()
            plt.xlim(t_1,t_2)
            f = model(time,start_1)
            plt.errorbar(lc_data[:,0], lc_data[:,1], yerr = lc_data[:,2])
            plt.title('Data and Fitted No MM Model (%s)' %case[z])
            plt.plot(time,f,'k-')
            plt.show()
            # pdf.savefig()
            # plt.close()
                    
            fig, axes = plt.subplots(n_dim, figsize=(10, 7), sharex=True)
            
            #parameters to fit
            samples_plot = sampler.get_chain()
            labels = parameters_to_fit
            #plt.title("Parameters to fit %s" %case[z])
            for i in range(n_dim):
                ax = axes[i]
                ax.plot(samples_plot[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples_plot))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                
            axes[-1].set_xlabel("step number");
            #plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/test1/params_%s.png" %case[z])
            plt.show()
            # pdf.savefig()
            # plt.close()
            
        case_stats()
        case_info()
    

                
                        