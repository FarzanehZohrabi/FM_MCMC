#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 07:04:27 2022

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

case = '104'


#likelyhood functions are in both linear and logarithmic parameters
def ln_like(theta, t, f, f_err):
    t_0, t_E, u_0, F0, fs = theta
    u = (u_0**2 + ((t-t_0)/t_E)**2)**0.5
    #if u == 0:
    #    u == exp(u)
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
    #if u == 0:
    #    u == exp(u)
    A_t = (u**2 + 2)/(u*(u**2+4)**0.5)
    Fs = fs*F0; Fb = F0*(1-fs)
    F_t = Fs*A_t + Fb
    model = F_t
    sigma2 = f_err ** 2
    return -0.5 * np.sum((f - model)**2/sigma2)


def ln_prior_log(theta):
    t_0, log_tE, log_u0, F0, log_fs = theta
    if -5 < log_fs < 0.1:
        return -np.inf
    if -2.0 < log_tE < 4.0:
        return -np.inf 
    if -4.0 < log_u0 < 1.0:
        return -np.inf 
    return 0.0

def ln_prior(theta):
    t_0, t_E, u_0, F0, fs = theta
    return 0.0

def ln_prob(theta, t, f, f_err):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnprobval = lp + ln_like(theta, t, f, f_err)
    return lnprobval

def ln_prob_log(theta, t, f, f_err):
    lp = ln_prior_log(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnprobval = lp + ln_like_log(theta, t, f, f_err)
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

# functions to compute major and minor axes
# 68% contained = chi-squared val of 2.28
def maj_axes(n):
    return np.sqrt(2.28*max(n),dtype=float)
def min_axes(n):
    return np.sqrt(2.28*min(n),dtype=float)

theta_grid = np.linspace(0,2*np.pi)
parameters_lin = ["t_0", "t_E", "u_0", "F0", "fs"]
image_no=1


n=int(5)
def areSame(cij_inv,cij_inv1):

   for i in range(n):
      for j in range(n):
         if (cij_inv[i][j] !=  cij_inv1[i][j]):
            return 0
   return 1

def corner_plt_inv():
    plt.figure()
    fig = corner.corner(samples, labels = parameters_lin,levels = (0.68, 0.95, 0.99))
 

    plt.title('Corner Plot %s (inv)' %case)
    axs = np.array(fig.axes)
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
    plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/SL_corner_plots/SL_%s_inv_corner.jpg" %case)
    #plt.show()

def corner_plt_MCMC():
    plt.figure()
    fig = corner.corner(samples, labels = parameters_lin,levels = (0.68, 0.95, 0.99))
    
    plt.title('Corner Plot %s (chains)' %case)
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
    plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/SL_corner_plots/SL_%s_MCMC_corner.jpg" %case)
    #plt.show() 


def inv_MCMC_plt():
    plt.figure()
    fig = corner.corner(samples, labels = parameters_lin,levels = (0.68, 0.95, 0.99))
    
    plt.title('Corner Plot %s (chains) & (inv)' %case)
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
    plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/SL_corner_plots/SL_%s_inv_MCMC_corner.jpg" %case)
    #plt.show()
    
def lin_model(t, theta):
    t_0, tE, u0, F0, fs = theta
    u = (u_0**2 + ((t-t_0)/t_E)**2)**0.5
    A_t = (u**2 + 2)/(u*np.sqrt(u**2+4))
    Fs = fs*F0
    Fb = F0*(1-fs)
    F_t = Fs*A_t + Fb
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
    F_t = Fs*A_t + Fb
    no_mm_model = F_t
    return no_mm_model

#MM Magnification Modeling
def MM_lin_model(t, theta):
    t_0, tE, u0, F0, fs = theta
    time = np.arange(t_1, t_2, 0.001)
    PS_model = mm.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E})
    fs = start_1[-1]
    F0 = start_1[-2]
    Fs = fs*F0
    Fb = F0*(1-fs)
    F_t = Fs*PS_model.magnification(time) + Fb
    return F_t

def MM_log_model(t, theta):
    t_0, log_tE, log_u0, F0, log_fs = theta
    u_0 = 10**(log_u0)
    t_E = 10**(log_tE)
    fs = 10**(log_fs)
    time = np.arange(t_1, t_2, 0.001)
    PS_model = mm.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E})
    fs = start_1[-1]
    F0 = start_1[-2]
    Fs = fs*F0
    Fb = F0*(1-fs)
    F_t = Fs*PS_model.magnification(time) + Fb
    return F_t

# directing the data to a txt file
def txt_case():
    with open("/Users/farzanehzohrabi/Documents/MCMC/SL_Txt_Files/SL_%s.txt" %case, "a") as f:
        if u_0 > sigma_u0:
            print("linear case", file=f)
        if u_0 < sigma_u0:
            print("log case", file=f)
        if sigma_t0 < 0.1:
            print("well-constrained case", file=f)
        if sigma_t0 > 0.1:
            print("poorly-constrained case", file=f)
        #Fish = bij[:, np.newaxis]
        new_column = np.arange(5)
        Fisher = np.insert(bij, 0, new_column, axis=1)
        #bij = bij[:, np.newaxis]
        print("bij:", file=f)
        for row in Fisher:
            f.write(" ".join(str(item) for item in row) + "\n")
        print("cij (from inverting bij):", file=f)
        cij_fm = np.insert(cij_inv, 0, new_column, axis=1)
        for row in cij_fm:
            f.write(" ".join(str(item) for item in row) + "\n")  
        print("cij (from chains):", file=f)
        cij_MCMC = np.insert(cij_chains, 0, new_column, axis=1)
        for row in cij_MCMC:
            f.write(" ".join(str(item) for item in row) + "\n")        
        print("ratio of cij's (cij_inv/cij_chains):", file=f)
        ratio = np.insert(cij_ratio, 0, new_column, axis=1)
        for row in cij_MCMC:
            f.write(" ".join(str(item) for item in row) + "\n")        
        print("Fitted parameters:", file=f)
        for i in range(n_dim):
            r = results[1, i]
            print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]), file=f)
            
        print("All parameters in Fisher Matrix", file=f)
        fm_params = params_fm[1::]
        result_MCMC= []
        for i in range(n_dim):
            r = results[1, i]
            result = [r, results[2, i]-r, r-results[0, i]]
            result_MCMC.append(result)
        result_MCMC = np.array(result_MCMC)
        #MCMC_Measurments= np.insert("%s " %case, 0, np.arange(1))
        f.write(" ".join(np.insert("%s" %case, 0, np.arange(1)))+ " ")
        #f.write("%s " %case)
        if u_0 > sigma_u0 and sigma_t0 < 0.1 :
            f.write("linear_case well-constrained ")
        if u_0 > sigma_u0 and sigma_t0 > 0.1 :
            f.write("linear_case poorly-constrained ")
        if u_0 < sigma_u0 and sigma_t0 < 0.1:
            f.write("logarithmic_case well-constrained ")
        if u_0 < sigma_u0 and sigma_t0 > 0.1:
            f.write("logarithmic_case poorly-constrained ")
        for row in fm_params:
            f.write(" ".join(str(item) for item in row)+ " ")
        for j in result_MCMC:
                f.write(" ".join(str(item) for item in j)+ " ")

            
        #f.write(" ".join(str(params_fm)))

def stat_per_event():
    with open("/Users/farzanehzohrabi/Documents/MCMC/SL_Txt_Files/SL_%s.txt" %case,"r") as txt:
        txt = txt.read()
    txt = txt.splitlines()
    txt = pd.DataFrame({'name':txt}) # converting list into data frame
    txt = txt.name.astype(str).str.split(expand=True)
    x = len(txt.index)
    params1 = txt[x-5::]
    params1 = params1.to_numpy()
    params1 = params1[:,0:3]
    params1 = params1.astype(float)
    # t_E = params[t_E]
    
    # if params[1,1] > params[1,2]:
    #     sig_tE = params[1,1]
    # else:
    #     sig_tE = params[1,2]   
    # with open("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82_%s.det.fm.0" %case,"r") as fm:
    #     data = fm.read()
    # data = data.splitlines() # importing the fm.0 file to be read
    
    # df = pd.DataFrame({'name':data}) # converting list into data frame
    # df = df.name.str.split(expand=True)
    # df = df[25::] # last three rows with the important parameters 
    # params_fm = df.to_numpy()
    
    # tE_fm = float(params_fm[1,1])
    # sig_tE_fm = float(params_fm[2,1])
    # u0_fm = float(params_fm[1,2])
    # sig_u0_fm = float(params_fm[2,2])
    # with open("/Users/farzanehzohrabi/Documents/MCMC/SL_Txt_Files/SL_PE_stats.txt", "a") as f:
    #     print('%s (single lens)' %case, file=f)
    
def stat():
    with open("/Users/farzanehzohrabi/Documents/MCMC/SL_Txt_Files/SL_%s.txt" %case,"r") as txt:
        txt = txt.read()
    txt = txt.splitlines()
    txt = pd.DataFrame({'name':txt}) # converting list into data frame
    txt = txt.name.str.split(expand=True)
    x = len(txt.index)
    params1 = txt[x-5::]
    params1 = params1.to_numpy()
    params1 = params1[:,0:3]
    params1 = params1.astype(float)
    t_E = params1[1,0]
    
    if params1[1,1] > params1[1,2]:
        sig_tE = params1[1,1]
    else:
        sig_tE = params1[1,2]
        
    with open("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82_%s.det.fm.0" %case,"r") as fm:
        data = fm.read()
    data = data.splitlines() # importing the fm.0 file to be read
    
    df = pd.DataFrame({'name':data}) # converting list into data frame
    df = df.name.str.split(expand=True)
    df = df[25::] # last three rows with the important parameters 
    params_fm = df.to_numpy()
    
    tE_fm = float(params_fm[1,1])
    sig_tE_fm = float(params_fm[2,1])
    u0_fm = float(params_fm[1,2])
    sig_u0_fm = float(params_fm[2,2])
    
    with open("/Users/farzanehzohrabi/Documents/MCMC/SL_Txt_Files/SL_stats.txt", "a") as f:
        print('%s (single lens)' %case, file=f)
        
        # linear case: uncertainty is low   
        if abs(u0_fm) > sig_u0_fm:
            print("Linear Case", file=f)
            if 0.9*sig_tE < sig_tE_fm < 1.1*sig_tE:
                print("Both methods agree", file=f)
            else: print("The methods disagree", file=f)
            if sig_tE_fm > sig_tE:
                print("FM says it's poorly constrained, but MCMC says it's well-constrained", file=f)
            else: print("FM says it's well-constrained, but MCMC says it's poorly constrained", file=f)
                
        # log case: uncertainty is high     
        if abs(u0_fm) < sig_u0_fm:
            print("Log Case", file=f)
            siglog_tE = (sig_tE/t_E)/np.log(10)
            siglog_tE_fm = (sig_tE_fm/tE_fm)/np.log(10)
            c = 1/(3*np.log(10))
            if siglog_tE_fm < c and siglog_tE < c:
                print("Both methods show the event is well-constrained", file=f)
            elif siglog_tE_fm > c and siglog_tE > c:
                print("Both methods show the event is poorly constrained", file=f)
            else: print("The methods disagree", file=f)
            if siglog_tE_fm > siglog_tE:
                print("FM says it's poorly constrained, but MCMC says it's well-constrained", file=f)
            else: print("FM says it's well-constrained, but MCMC says it's poorly constrained", file=f)


