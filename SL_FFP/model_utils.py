import numpy as np
from PyAstronomy import pyasl
import emcee

VBM = None

def set_vbm(vbm_inst):
    """Call this once, under __main__, after you create & warm up VBMicrolensing."""
    global VBM
    VBM = vbm_inst
    
def ln_prior(theta, parameters_to_fit, finite_source):
    pt = dict(zip(parameters_to_fit, theta))

    #if 'log_fs' in pt:
        #if not (-5.0 < pt['log_fs'] < 0.1):
       #     return -np.inf

    #if 'log_tE' in pt:
     #   if not (-2.0 < pt['log_tE'] < 4.0):
      #      return -np.inf


    if finite_source:
        # log‐space ρ
        if 'log_rho' in pt:
            if not (-4.0 < pt['log_rho'] < 2.0):
                return -np.inf
        # linear ρ
        elif 'rho' in pt:
            if not (1e-4 < pt['rho'] < 1e2):
                return -np.inf
    elif not finite_source and 'log_u0' in pt:
        if not (-4.0 < pt['log_u0'] < 1.0):
            return -np.inf
        
    return 0.0

def ln_likelihood(theta, t, f, f_err, coords,
                  parameters_to_fit, finite_source,linear_case):

    pt = dict(zip(parameters_to_fit, theta))
    t0 = pt['t0']
    tE = 10**pt['log_tE'] if 'log_tE' in pt else pt['tE']
    u0 = 10**abs(pt['u0']) if 'log_u0' in pt else abs(pt['u0'])
    rho = (10**pt['log_rho'] if 'log_rho' in pt else pt.get('rho', None)) \
          if finite_source else None
    F0 = pt['F0'] 
    fs =10**pt['log_fs'] if 'log_fs' in pt else pt['fs']


    VBM.RelTol=1e-03
    VBM.Tol=1e-05
    
    if finite_source:
        LC = VBM.ESPLLightCurve([np.log(u0), np.log(tE) ,t0, np.log(rho)], t)
        A = LC[0]
    else:
        LC = VBM.PSPLLightCurve([np.log(u0), np.log(tE), t0], t)
        A = LC[0]
    
    A = np.asarray(A)

    F_t = fs * F0 * A + (1 - fs) * F0
    inv_var = 1.0 / f_err**2
    return -0.5 * np.sum((f - F_t)**2 * inv_var)



def ln_probability(theta, t, F, F_err, coords,parameters_to_fit, finite_source: bool,linear_case: bool):

	lp = ln_prior(theta,parameters_to_fit,finite_source)
	if not np.isfinite(lp):
		return -np.inf
	return lp + ln_likelihood(theta, t, F, F_err, coords,
                  parameters_to_fit, finite_source,linear_case)

def func(theta, t, F, F_err):
	try:
		lnprobval = -2. * ln_probability(theta, t, F, F_err, coords,parameters_to_fit, finite_source,linear_case)
	except ValueError: # NaN value case
		lnprobval = -np.inf # just set to negative infinity 
	return lnprobval

def uncertainties(percentiles):
	# percentiles = [15.8, 50, 84.1]
	med = percentiles[1]
	return med, percentiles[2] - med, med - percentiles[0]
    
def initialize_emcee(parameters_to_fit,
                     params,
                     sigmas_fm,
                     ra,
                     dec, finite_source, linear_case: bool,
                     n_walkers: int = 100,
                     n_steps: int = 3000,
                     n_burn: int = 500,
                     thin: int = 20):

	coords = pyasl.coordsDegToSexa(ra, dec)
	if not finite_source:
		# 'rho' is the 4th element (index 3)
		sigmas_fm.pop(3)
	sigmas = [ s * 1e-5 for s in sigmas_fm ]
	"""if not linear_case:
		if sigmas_fm[1] > 1:
			sigmas[1] = 0.3
		if sigmas_fm[3] > 1:
			sigmas[3] = 0.3            
		if sigmas_fm[5] > 1:
			sigmas[5] = 0.3
		if not finite_source:
			if sigmas_fm[2] > 1:
				sigmas[2] = 0.3 """
	n_dim = len(parameters_to_fit)
	# Starting point per walker
	start_values = [params[p] for p in parameters_to_fit]
	start_positions = [start_values + np.random.randn(n_dim) * sigmas
                       for _ in range(n_walkers)]
	return n_dim, n_walkers, n_steps, n_burn, thin, start_positions, start_values, coords


def run_emcee(ln_prob_fn,
               start_positions: list,
               n_walkers: int,
               n_steps: int,
               args: tuple = ()):  

    sampler = emcee.EnsembleSampler(n_walkers,
                                    len(start_positions[0]),
                                    ln_prob_fn,
                                    args=args)
    sampler.run_mcmc(start_positions, n_steps, progress=True)
    return sampler


def process_emcee_results(sampler,
                          n_burn: int,
                          thin: int) -> tuple:

    flat = sampler.get_chain(discard=n_burn, flat=True, thin=thin)
    samples = flat.reshape((-1, flat.shape[-1]))
    # Covariance from samples
    cov_chains = np.cov(samples.T)
    # Percentiles: 15.8, 50, 84.1
    results = np.percentile(samples, [15.8, 50.0, 84.1], axis=0)
    p15, p50, p84 = results
    sigma_plus = p84 - p50
    sigma_minus = p50 - p15
    sigma_mcmc = 0.5*(sigma_plus + sigma_minus)
    tE_MC = p50[1]
    sig_tE_MC = sigma_mcmc[1]
    return samples, cov_chains, results, sigma_mcmc, tE_MC, sig_tE_MC
    
		
