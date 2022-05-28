#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:40:04 2021

@author: emersongehr

modeling single lens events -- no MulensModel
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

# enter the number of the microlensing event: only thing that should need changing between cases
case = '104'

with open("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82_%s.det.fm.0" %case,"r") as fm:
    data = fm.read()
data = data.splitlines() # importing the fm.0 file to be read

df = pd.DataFrame({'name':data}) # converting list into data frame
df = df.name.str.split(expand=True)
df_bij = df[1:7]
bij = df_bij.to_numpy(dtype=float)
df = df[25::] # last three rows with the important parameters 
params_fm = df.to_numpy()

# Finding the data in the directory
with open("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82_%s.det.lc" %case,"r") as lc:
    lc_data = lc.read()
lc_data = lc_data.splitlines()
lc = pd.DataFrame({'name':lc_data}) # converting list into data frame
lc = lc.name.str.split(expand=True)
lc_file = lc.to_numpy()
lc_data = lc[8::]
lc_data = lc_data.to_numpy(dtype=float)
lc_data = lc_data[:,0:3]

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

od = pd.DataFrame(np.loadtxt("/Users/farzanehzohrabi/Documents/MCMC/singleLens/singleLens_0_82.out", dtype=str))
out_data = od.to_numpy()
out_data = out_data[:,0:7]
ra = float(out_data[int(case),5])
dec = float(out_data[int(case),6])
sexa = pyasl.coordsDegToSexa(ra, dec)
coords = sexa

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

sigmas = [float(params_fm[2,0]), float(params_fm[2,1]), float(params_fm[2,2]),
              float(params_fm[2,4]), float(params_fm[2,5])]
              
bij_cols = np.column_stack((bij[0:,:3], bij[0:,4:]))
x = bij_cols[0:3,:]
y = bij_cols[4:,:]
bij = np.row_stack((x,y)) # bij with rs removed

# account for abs(u_0): multiply all u_0 by -1
if u_0 < 0:
    bij[:,2] = -bij[:,2]
    bij[2,:] = -bij[2,:]
    u_0 = abs(u_0)

if u_0 < sigma_u0:
    print('log case')
    # Which parameters we want to fit?
    parameters_to_fit = ["t_0", "logt_E", "logu_0", "F0", "log_fs"] 
                 
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
    
    cij_inv = np.linalg.inv(bij) # 5x5 covariance matrix

if u_0 > sigma_u0:
    print('linear case')
    
    # Which parameters we want to fit?
    parameters_to_fit = ["t_0", "t_E", "u_0", "F0", "fs"]
    
    sigmas = [float(params_fm[2,0]), float(params_fm[2,1]), float(params_fm[2,2]),
              float(params_fm[2,4]), float(params_fm[2,5])]
    
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
    
    cij_inv = np.linalg.inv(bij) # 5x5 covariance matrix with rs removed

chains_tr = np.transpose(samples)
cij_chains= np.cov(chains_tr) 

# ratio of cij_inv/cij_chains
cij_ratio = cij_inv/cij_chains

# Results:
results = np.percentile(samples, [16, 50, 84], axis=0)
# axis=0 goes down the columns
print("Fitted parameters:")
for i in range(n_dim):
    r = results[1, i]
    print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]))
    
# functions to compute major and minor axes
# 68% contained = chi-squared val of 2.28
def maj_axes(n):
    return np.sqrt(2.28*max(n),dtype=float)
def min_axes(n):
    return np.sqrt(2.28*min(n),dtype=float)

theta_grid = np.linspace(0,2*np.pi)
parameters_lin = ["t_0", "t_E", "u_0", "F0", "fs"]

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
            #plt.savefig("/Users/emersongehr/Desktop/SL_graphs/SL_%s_cov_ch_%s.jpg" %(case, str(image_no)))
            image_no = image_no + 1

cov_all_ch = cov_all_ch[1:,:]

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
            # plt.xlabel(parameters_lin[i])
            # plt.ylabel(parameters_lin[j])
            #plt.savefig("/Users/emersongehr/Desktop/SL_graphs/SL_%s_cov_inv_%s.jpg" %(case, str(image_no)))
            image_no = image_no + 1
           
                
cov_all_inv = cov_all_inv[1:,:]

cov_all_ch = np.delete(cov_all_ch, (8,9,16,17,18,19,24,25,26,27,28,29), axis=0)
cov_all_inv = np.delete(cov_all_inv, (8,9,16,17,18,19,24,25,26,27,28,29), axis=0) 

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
#plt.savefig("/Users/emersongehr/Desktop/SL_graphs/SL_%s_ch_corner.jpg" %case)
plt.show() 

#plt.figure()
#fig = corner.corner(samples, labels = parameters_lin,levels = (0.68, 0.95, 0.99))

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
#plt.savefig("/Users/emersongehr/Desktop/SL_graphs/SL_%s_inv_corner.jpg" %case)
plt.show() 


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

t_1 = t_0 - 70
t_2 = t_0 + 70

# plotting the data by itself
plt.figure()
plt.plot(lc_data[:,0], -m, 'ok',markersize=3) # plotting magnitude v time
plt.errorbar(lc_data[:,0], -m, sigma_m, fmt='None', ecolor=('k'), elinewidth=(1), 
            capsize=(2))
plt.title("%s" %case)
plt.xlim(t_1,t_2)

# tr_time_mag = np.transpose(time_mag)
# MM_mag_data = mm.MulensData(tr_time_mag)
# my_model = mm.Model(mm_params, coords=coords)
# my_event = mm.Event(datasets=MM_mag_data, model=my_model)

# # generate fitted models
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
# plt.savefig("/Users/farzanehzohrabi/Documents/MCMC/lc_generating_codes/SL_%s_models.jpg" %case)

def ln_like_model(theta, t):
    t_0, log_tE, log_u0, F0, fs = theta
    t_E = 10**(log_tE)
    u_0 = 10**(log_u0)
    u = (u_0**2 + ((t-t_0)/t_E)**2)**0.5
    A_t = (u**2 + 2)/(u*(u**2+4)**0.5)
    Fs = fs*F0; Fb = F0*(1-fs)
    F_t = Fs*A_t + Fb
    model = F_t
    return model

plt.figure()
plt.xlim(t_1,t_2)
t = np.arange(t_1, t_2, 0.001)
f = ln_like_model(start_1, t)
plt.errorbar(lc_data[:,0], lc_data[:,1], yerr = lc_data[:,2])
plt.plot(t,f,'k-')

# directing the data to a txt file
with open("/Users/farzanehzohrabi/Documents/MCMC/SL_Txt_Files/SL_%s.txt" %case, "a") as f:
    if u_0 > sigma_u0:
        print("linear case", file=f)
    if u_0 < sigma_u0:
        print("log case", file=f)
    if sigma_t0 < 0.1:
        print("well-constrained case", file=f)
    if sigma_t0 > 0.1:
        print("poorly-constrained case", file=f)
    print("bij:", bij, file=f)  
    print("cij (from inverting bij):", cij_inv, file=f)
    print("cij (from chains):", cij_chains, file=f)
    print("ratio of cij's (cij_inv/cij_chains):", cij_ratio, file=f)
    print("Fitted parameters:", file=f)
    for i in range(n_dim):
        r = results[1, i]
        print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]), file=f)

   