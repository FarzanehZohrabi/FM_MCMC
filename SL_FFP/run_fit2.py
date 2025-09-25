#!/usr/bin/env python3
import multiprocessing as mp

import faulthandler


import os
import numpy as np
import pandas as pd
import emcee
from matplotlib.backends.backend_pdf import PdfPages

# now it’s safe to import modules that import multiprocessing or VBMicrolensing
import VBMicrolensing
import model_utils
from data_utils import (
    read_outfile, check_linear_case, check_finite_source,
    pick_parameters_to_fit, cov_jacobian, read_fm, sig_fm, read_lc
)
from plot_utils import (
    plot_corner_with_ellipses, plot_parameter_evolution, plot_lc
)
from stats_utils import case_stats, case_info

from scipy.optimize import minimize


class fitcomp:

    def __init__(self):
        vbm_inst = VBMicrolensing.VBMicrolensing()
        _ = vbm_inst.ESPLMag2(1.0, 1e-3)
        model_utils.set_vbm(vbm_inst)
        mp.set_start_method('spawn', force=True)
        faulthandler.enable(all_threads=True)

        self.n_walkers=30
        self.n_steps=10000
        self.n_burn=1000
        self.thin=3

        self.processing_done = False

    def set_mcmc_params(self,n_walkers=30, n_steps=10000, n_burn=1000, thin=5):
            self.n_walkers=n_walkers
            self.n_steps=n_steps
            self.n_burn=n_burn
            self.thin=thin
    
    
    def run_case(self,input_directory, out_file, outdir, obsgroup, match,case_id):
        self.df, self.base = read_outfile(out_file)
        self.input_directory = input_directory
        self.outdir = outdir
        self.obsgroup = obsgroup
        self.case_id = case_id
        self.match = match
        #self.case_ids = self.df['EventID'].astype(int).tolist()

        # Only process a specific case (remove or adjust this filter as needed)
        #if case_id != 2:
        #    continue
        
        self.row = self.df[self.df['EventID'] == self.case_id].iloc[0]
        self.dchi2 = self.row[f"ObsGroup_{match}_chi2"]
        #if dchi2 < 300.0:
        #    continue
        
        # Coordinates
        self.ra = self.row['ra_deg']
        self.dec = self.row['dec_deg']
        
        # Extract key parameters
        self.t0 = self.row['t0lens1']
        self.tE = self.row['tE_ref']
        self.u0 = self.row['u0lens1']
        self.rho = self.row['rho']
        self.F0 = 1.0
        self.fs = self.row[f"Obs_{match}_fs"]

        self.params_gulls = [self.t0, self.tE, self.u0, self.rho, self.F0, self.fs]
        #if self.tE < 0 or self.rho < 0:
        #    print(f"Skipping Event {case_id}: invalid parameters {params_gulls}")
        #    continue

        # File paths
        self.fm_file = os.path.join(self.input_directory, f"{self.base}_{self.case_id}.det.fm.0")
        self.lc_file = os.path.join(self.input_directory, f"{self.base}_{self.case_id}.det.lc")
        if not os.path.exists(self.fm_file) or not os.path.exists(self.lc_file):
            print(f"Error: fisher matrix file ({self.fm_file}) or lightcurve file ({self.lc_file}) don't exist.")
            return
        
        # Fisher‐matrix sigmas
        self.sigmas_fm = sig_fm(self.fm_file)
        sigma_t0_fm, sigma_tE_fm, sigma_u0_fm, sigma_rho_fm, sigma_F0_fm, sigma_fs_fm = self.sigmas_fm
        
        self.finite_source = check_finite_source(self.rho, self.u0, sigma_rho_fm, threshold=1.0/3)
        self.linear_case   = check_linear_case (sigma_u0_fm, self.u0, threshold=1.0/3)
        self.parameters_to_fit, self.params = pick_parameters_to_fit(
            self.t0, self.tE, self.u0, self.rho, self.F0, self.fs, self.finite_source, self.linear_case
        )
        print(self.params)
        #if finite_source and not linear_case:
        #           print("case_id", "chi2", "lc path")
        #            print(case_id, dchi2, lc_file)
        #            continue
        #else:
        #            continue
        # Read light curve
        self.data_df, self.fsm, self.t, self.F, self.F_err = read_lc(self.lc_file)

        # Covariance from Fisher matrix
        self.bij, self.u0, self.cov_fm, self.sigmas_fm = read_fm(self.fm_file, self.rho, self.u0,
                                                                  self.finite_source)
        if not self.linear_case:
            self.cov_fm, self.sigmas_fm = cov_jacobian(
                self.parameters_to_fit, self.cov_fm, self.tE, self.u0, self.rho, self.fs,
                self.finite_source
            )

        # Set up and run MCMC
        self.ln_prob_fn = model_utils.ln_probability
        self.n_dim, self.n_walkers, self.n_steps, self.n_burn, self.thin, self.start_positions, self.start_vals, self.coords = \
                model_utils.initialize_emcee(
                    self.parameters_to_fit, self.params, self.sigmas_fm,
                    self.ra, self.dec, self.finite_source,self.linear_case,
                    n_walkers=self.n_walkers, n_steps=self.n_steps, n_burn=self.n_burn, thin=self.thin
                )
            #n_walkers=30, n_steps=4000, n_burn=500, thin=20

            
        emcee_args = (self.t, self.F, self.F_err, self.coords, self.parameters_to_fit, self.finite_source, self.linear_case)
        self.sampler = model_utils.run_emcee(
                self.ln_prob_fn, self.start_positions, self.n_walkers, self.n_steps, args=emcee_args
            )
        result = model_utils.process_emcee_results(self.sampler, self.n_burn, self.thin)
        self.samples, self.cov_mcmc, self.results, self.sigma_mcmc, self.tE_MC, self.sig_tE_MC = result

        self.processing_done = True
        print(f"Processing Event {self.case_id} complete.")


    def run_output(self):

        if not self.processing_done:
            print("run_fit ")

        # Generate PDF report
        pdf_path = os.path.join(self.outdir, f"{self.base}_{self.case_id}.pdf")
        print(pdf_path)
        
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            print("here")
            plot_corner_with_ellipses(self.samples, self.parameters_to_fit, self.cov_mcmc,
                                      self.cov_fm, self.params_gulls, self.sigmas_fm,
                                      self.sigma_mcmc, self.finite_source, pdf)
            plot_lc(self.t0, self.tE, self.data_df, self.fsm, match=self.match, nobs=len(self.obsgroup), displayobs=None, obsgroup=self.obsgroup, models=None, output_pdf=pdf)
            plot_parameter_evolution(self.sampler, self.parameters_to_fit, pdf)
            
        # Save summary and details
        case_stats(self.case_id, self.results, self.sigmas_fm, self.params_gulls, self.tE,
                   self.tE_MC, self.sig_tE_MC, self.outdir, self.base, self.finite_source,
                   self.linear_case)
        case_info(self.case_id, self.bij, self.sigmas_fm, self.cov_fm, self.results, self.cov_mcmc, self.outdir, self.base, self.finite_source, self.linear_case)







#def init():
#    vbm_inst = VBMicrolensing.VBMicrolensing()
#    _ = vbm_inst.ESPLMag2(1.0, 1e-3)
#    model_utils.set_vbm(vbm_inst)

#if __name__ == '__main__':
#    main()
