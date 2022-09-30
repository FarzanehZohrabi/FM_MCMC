#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:28:47 2022

@author: farzanehzohrabi
"""

"""
Fits PSPL model with parallax using EMCEE sampler.
"""
import os
import sys
import pandas as pd
import numpy as np
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
try:
    import emcee
except ImportError as err:
    print(err)
    print("\nEMCEE could not be imported.")
    print("Get it from: http://dfm.io/emcee/current/user/install/")
    print("and re-run the script")
    sys.exit(1)
import matplotlib.pyplot as plt

import MulensModel as mm

od = pd.read_csv("/Users/farzanehzohrabi/Documents/MCMC/kmt_MLE.txt", sep=" ",on_bad_lines='skip')
file_name= od.iloc[:,[0,9]]
event_duration= od.iloc[:,[9]].astype(float)

par_case=[]

for i in range(len(file_name)):
    if od.iloc[i,9] > 60.0 :
        par_case.append(od.iloc[i,0])
        par_case.append(od.iloc[i,9])


case="0307"
kmt = pd.read_csv("/Users/farzanehzohrabi/opt/anaconda3/lib/python3.8/site-packages/MulensModel/data/photometry_files/kmt/kmt0307.txt", on_bad_lines= "skip", sep= "\s+")#, sep=" ",on_bad_lines='skip')
# hjd = kmt[0] #make a variable here
# mag_col= kmt[3]
# magerr= kmt[4]
phot_info = kmt.iloc[:,[0,3,4]].to_numpy().T

# with open("/Users/farzanehzohrabi/opt/anaconda3/lib/python3.8/site-packages/MulensModel/data/photometry_files/kmt/kmt0307_phot.txt", "a+") as f:
#     dfAsString = phot_info.to_string(header=False, index=False)
#     f.write(dfAsString)

# Define likelihood functions
def ln_like(theta, event, parameters_to_fit):
    """ likelihood function """
    for (parameter, value) in zip(parameters_to_fit, theta):
        setattr(event.model.parameters, parameter, value)

    chi2 = event.get_chi2()
    if chi2 < ln_like.best[0]:
        ln_like.best = [chi2, theta]
    return -0.5 * chi2
ln_like.best = [np.inf]


def ln_prior(theta, parameters_to_fit):
    """priors - we only reject obviously wrong models"""
    if theta[parameters_to_fit.index("t_E")] < 0.:
        return -np.inf
    return 0.0


def ln_prob(theta, event, parameters_to_fit):
    """ combines likelihood and priors"""
    ln_prior_ = ln_prior(theta, parameters_to_fit)
    if not np.isfinite(ln_prior_):
        return -np.inf
    ln_like_ = ln_like(theta, event, parameters_to_fit)

    # In the cases that source fluxes are negative we want to return
    # these as if they were not in priors.
    if np.isnan(ln_like_):
        return -np.inf

    return ln_prior_ + ln_like_


# Read the data
# file_name = os.path.join(
#     mm.DATA_PATH, "photometry_files", "kmt",
#     "kmt0307_phot.txt")
my_data = mm.MulensData(data_list=phot_info, add_2450000=True)

coords = "17:47:45.60 -26:26:23.57"

# Starting parameters:
params = dict()
params['t_0'] = 9325.86444 + 2450000
params['t_0_par'] =  9325.9 + 2450000
params['u_0'] = 0.200  # Change sign of u_0 to find the other solution.
params['t_E'] = 60.77
params['pi_E_N'] = 0.1
params['pi_E_E'] = 0.1
my_model = mm.Model(params, coords=coords)
my_event = mm.Event(datasets=my_data, model=my_model)

# Which parameters we want to fit?
parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
# And remember to provide dispersions to draw starting set of points
sigmas = [0.01, 0.001, 0.001, 0.01, 0.01]

# Initializations for EMCEE
n_dim = len(parameters_to_fit)
n_walkers = 40
n_steps = 500
n_burn = 150
# Including the set of n_walkers starting points:
start_1 = [params[p] for p in parameters_to_fit]
start = [start_1 + np.random.randn(n_dim) * sigmas
         for i in range(n_walkers)]

# Run emcee (this can take some time):
sampler = emcee.EnsembleSampler(
    n_walkers, n_dim, ln_prob, args=(my_event, parameters_to_fit))
sampler.run_mcmc(start, n_steps)

# Remove burn-in samples and reshape:
samples = sampler.chain[:, n_burn:, :].reshape((-1, n_dim))

# Results:
#results = np.percentile(samples, [16, 50, 84], axis=0)
#corner plot
chains_tr = np.transpose(samples)
cij_chains= np.cov(chains_tr) 
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

cov_all_ch = cov_all_ch[1:,:]

cov_all_ch = np.delete(cov_all_ch, (8,9,16,17,18,19,24,25,26,27,28,29), axis=0)

print("Fitted parameters:")
for i in range(n_dim):
    r = results[1, i]
    print("{:.5f} {:.5f} {:.5f}".format(r, results[2, i]-r, r-results[0, i]))

print("\nSmallest chi2 model:")
print(*[b if isinstance(b, float) else b.value for b in ln_like.best[1]])
print(ln_like.best[0])

# Now let's plot 3 models





pdf = PdfPages('/Users/farzanehzohrabi/Documents/MCMC/parfit/annpar_%s.pdf' %case)  
with pdf:
    model_0 = mm.Model({'t_0': params['t_0'], 'u_0': params['u_0'], 't_E': params['t_E']})
    model_1 = mm.Model(
        {'t_0': results[1,0], 'u_0': results[1,1], 't_E':results[1,2],
         'pi_E_N': results[1,3], 'pi_E_E': results[1,4], 't_0_par': params['t_0_par']},
        coords=coords)
    # model_2 = mm.Model(
    #     {'t_0': 2459420.7334996588, 'u_0': 0.008984947717384048, 't_E': 108.61010813539131,
    #      'pi_E_N': 0.16027728571103297, 'pi_E_E': -0.0689145509272246, 't_0_par': params['t_0_par']},
    #     coords=coords)
    event_0 = mm.Event(model=model_0, datasets=[my_data])
    event_1 = mm.Event(model=model_1, datasets=[my_data])
    # event_2 = mm.Event(model=model_2, datasets=[my_data])
    
    t_1 = 9325.86444 + 2450000 -200
    t_2 = 9325.86444 + 2450000 +200
    plot_params = {'lw': 2.5, 'alpha': 0.3, 'subtract_2450000': True,
                   't_start': t_1, 't_stop': t_2}
    
    my_event.plot_data(subtract_2450000=True)
    event_0.plot_model(label='no pi_E', **plot_params)
    event_1.plot_model(label='pi_E, u_0>0', **plot_params)
    # event_2.plot_model(
        # label='pi_E, u_0<0', color='black', ls='dashed', **plot_params)
    
    plt.xlim(t_1-2450000., t_2-2450000.)
    plt.ylim(22,17)
    plt.legend(loc='best')
    plt.title('Data and 3 fitted models')
    #plt.savefig('/Users/farzanehzohrabi/Documents/MCMC/parfit0307.png', dpi=600)
    pdf.savefig()
    plt.close()
    plt.show()
    
    plt.figure()
    fig = corner.corner(samples, labels = parameters_to_fit,levels = (0.68, 0.95, 0.99))
    plt.title('parallax %s' %case)
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
    pdf.savefig()
    plt.close()
    plt.show()
