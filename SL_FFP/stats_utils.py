import os
import numpy as np

def case_stats(case_id, results, sig_cov_fm, params_gulls, tE,tE_MC,sig_tE_MC, outdir,base,
               finite_source: bool, linear_case: bool):
    stat_file = os.path.join(outdir, f'{base}_stats.txt')
    with open(stat_file, 'a+') as stat:
        n_dim = results.shape[1]
        # build MCMC summary
        result_MCMC = np.array([[results[1,i],
                                  results[2,i]-results[1,i],
                                  results[1,i]-results[0,i]]
                                for i in range(n_dim)])
        t_E_MC = result_MCMC[1,0]
        sig_tE_MC = max(result_MCMC[1,1], result_MCMC[1,2])

        # header
        kind = 'linear_case' if linear_case else 'logarithmic_case'
        stat.write(f"{case_id} FFP {kind} ")

        # agreement logic
        if not linear_case:
            cond1 = sig_cov_fm[1] <= 1/3
            cond2 = sig_tE_MC <= 1/3
        else:
            cond1 = sig_cov_fm[1]/tE <= 1/3
            cond2 = sig_tE_MC/tE_MC <= 1/3

        if   cond1 and cond2: stat.write('well-constrained agree_well | ')
        elif cond1 and not cond2: stat.write('well-constrained disagree_fmw_MCp | ')
        elif not cond1 and not cond2: stat.write('poorly-constrained agree_poor | ')
        else: stat.write('poorly-constrained disagree_fmp_MCw | ')

        # write FM params (flat)
        fm_vals = np.round(params_gulls, 6)
        stat.write(' '.join(f"{v:.6f}" for v in fm_vals))
        stat.write(' | ')

        # write sig_calculated
        sig_vals = np.round(sig_cov_fm, 6)
        stat.write(' '.join(f"{v:.6f}" for v in sig_vals))
        stat.write(' | ')

        # write MCMC results
        mcmc_vals = np.round(result_MCMC, 6)
        # each row is [median, +err, -err]
        for med, up, down in mcmc_vals:
            stat.write(f"{med:.6f} {up:.6f} {down:.6f} ")
        stat.write('\n')



def case_info(case_id, bij, sig_cov_fm, cov_fm, results, cov_mcmc, outdir,base,finite_source: bool,linear_case: bool):
	info_file = os.path.join(outdir, f'{base}_{case_id}.txt')
	with open(info_file, 'a') as f:
		f.write('Finite source\n' if finite_source else 'point source\n')
		f.write('linear case\n' if linear_case else 'log case\n')
		f.write(('well-constrained case\n') if sig_cov_fm[1] <= 1/3 else 'poorly-constrained case\n')
		f.write(f"bij: {bij}\n")
		f.write(f"sigma FM calculated: {sig_cov_fm}\n")
		f.write(f"cij FM: {cov_fm}\n")
		f.write(f"cij MCMC: {cov_mcmc}\n")
		f.write(f"ratio cov_fm/cov_mcmc: {cov_fm/cov_mcmc}\n")
		f.write("Fitted parameters:\n")
		for i in range(results.shape[1]):
			r = results[1,i]
			f.write(f"{r:.5f} {results[2,i]-r:.5f} {r-results[0,i]:.5f}\n")