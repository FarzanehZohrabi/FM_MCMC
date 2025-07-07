#!/usr/bin/env python3
import multiprocessing as mp

mp.set_start_method('spawn', force=True)
import faulthandler
faulthandler.enable(all_threads=True)

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



def run_case(directory, out_file, outdir, obsgroup, match):
    df, base = read_outfile(out_file)
    case_ids = df['EventID'].astype(int).tolist()

    for case_id in case_ids:
        try:
            # Only process a specific case (remove or adjust this filter as needed)
            if case_id != 2:
                continue

            row = df[df['EventID'] == case_id].iloc[0]
            dchi2 = row[f"ObsGroup_{match}_chi2"]
            if dchi2 < 300.0:
                continue

            # Coordinates
            ra = row['ra_deg']
            dec = row['dec_deg']

            # Extract key parameters
            t0 = row['t0lens1']
            tE = row['tE_ref']
            u0 = row['u0lens1']
            rho = row['rho']
            F0 = 1.0
            fs = row[f"Obs_{match}_fs"]

            params_gulls = [t0, tE, u0, rho, F0, fs]
            if tE < 0 or rho < 0:
                print(f"Skipping Event {case_id}: invalid parameters {params_gulls}")
                continue

            # File paths
            fm_file = os.path.join(directory, f"{base}_{case_id}.det.fm.0")
            lc_file = os.path.join(directory, f"{base}_{case_id}.det.lc")
            if not os.path.exists(fm_file) or not os.path.exists(lc_file):
                continue

            # Fisher‐matrix sigmas
            sigmas_fm = sig_fm(fm_file)
            sigma_t0_fm, sigma_tE_fm, sigma_u0_fm, sigma_rho_fm, sigma_F0_fm, sigma_fs_fm = sigmas_fm

            finite_source = check_finite_source(rho, u0, sigma_rho_fm, threshold=1/3)
            linear_case   = check_linear_case (sigma_u0_fm, u0, threshold=1/3)
            parameters_to_fit, params = pick_parameters_to_fit(
                t0, tE, u0, rho, F0, fs, finite_source, linear_case
            )
            print(params)
            #if finite_source and not linear_case:
             #           print("case_id", "chi2", "lc path")
            #            print(case_id, dchi2, lc_file)
            #            continue
            #else:
            #            continue
            # Read light curve
            data_df, fsm, t, F, F_err = read_lc(lc_file)

            # Covariance from Fisher matrix
            bij, u0, cov_fm, sig_cov_fm = read_fm(fm_file, rho, u0, finite_source)
            if not linear_case:
                cov_fm, sig_cov_fm = cov_jacobian(
                    parameters_to_fit, cov_fm, tE, u0, rho, fs, finite_source
                )

            # Set up and run MCMC
            ln_prob_fn = model_utils.ln_probability
            n_dim, n_walkers, n_steps, n_burn, thin, start_positions, start_vals, coords = \
                model_utils.initialize_emcee(
                    parameters_to_fit, params, sigmas_fm,
                    ra, dec, finite_source,linear_case,
                    n_walkers=30, n_steps=2000, n_burn=50, thin=20
                )
            emcee_args = (t, F, F_err, coords, parameters_to_fit, finite_source, linear_case)
            sampler = model_utils.run_emcee(
                ln_prob_fn, start_positions, n_walkers, n_steps, args=emcee_args
            )
            samples, cov_mcmc, results, sigma_mcmc, tE_MC, sig_tE_MC = model_utils.process_emcee_results(
                sampler, n_burn, thin
            )

            # Generate PDF report
            pdf_path = os.path.join(directory, f"{base}_{case_id}.pdf")
         
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with PdfPages(pdf_path) as pdf:
                plot_corner_with_ellipses(samples, parameters_to_fit, cov_mcmc, cov_fm,params_gulls,sigmas_fm,sigma_mcmc,finite_source, pdf)
                plot_lc(t0, tE, data_df, fsm, match=match, nobs=3, displayobs=None, obsgroup=obsgroup, models=None, output_pdf=pdf)
                plot_parameter_evolution(sampler, parameters_to_fit, pdf)

            # Save summary and details
            case_stats(case_id, results, sig_cov_fm, params_gulls, tE,tE_MC,sig_tE_MC, outdir, base, finite_source, linear_case)
            case_info(case_id, bij, sig_cov_fm, cov_fm, results, cov_mcmc, outdir, base, finite_source, linear_case)

            print(f"Processing Event {case_id} complete.")

        except Exception as e:
            print(f"Skipping Event {case_id} due to error: {e}")
            continue




def main():
	vbm_inst = VBMicrolensing.VBMicrolensing()
	_ = vbm_inst.ESPLMag2(1.0, 1e-3)
	model_utils.set_vbm(vbm_inst)
	directory = "/Users/ffp_fish_overguide/"
	out_file = os.path.join(directory, 'ffp_fish_overguide_0_52.out')
	outdir = "/Users/FFP_FM_MCMC"
	obsgroup = ["F146", "F087", "F213"]
	match = 0 #ref observatory
	run_case(directory, out_file,outdir,obsgroup, match=match)

if __name__ == '__main__':
    main()