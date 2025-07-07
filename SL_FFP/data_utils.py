import os
import glob
import numpy as np
import pandas as pd
from PyAstronomy import pyasl


def read_outfile(out_file: str) -> pd.DataFrame:
    df = pd.read_csv(out_file, delim_whitespace=True, comment="#")
    base = os.path.splitext(os.path.basename(out_file))[0]
    return df , base

def check_linear_case(sigma_u0_fm, u0, threshold = 1/3) -> bool:

    return sigma_u0_fm < threshold * abs(u0)

def check_finite_source(rho, u0, sigma_rho_fm, threshold = 1/3) -> bool:
	return (rho > abs(u0)) and (sigma_rho_fm <= threshold * rho)
	
def pick_parameters_to_fit(t0,tE, u0, rho,F0, fs,finite_source,linear_case):
	param_orders = {
    		(True,  True ): ['t0','tE','u0','rho','F0','fs'],
    		(True,  False): ['t0','log_tE','u0','log_rho','F0','log_fs'],
    		(False, True ): ['t0','tE','u0','F0','fs'],
    		(False, False): ['t0','log_tE','log_u0','F0','log_fs'],
		}
	raw = dict(t0=t0, tE=tE, u0=abs(u0),
			rho=rho, F0=F0, fs=fs)

	parameters_to_fit = param_orders[(finite_source, linear_case)]

	# build a dict mapping each parameter name → its initial value
	params0 = {}
	for name in parameters_to_fit:
		if name.startswith("log_"):
			base = name.split("log_")[1]   # tE, u0 or rho or fs
			params0[name] = np.log10(raw[base])
		else:
			params0[name] = raw[name]

	#print("parameters_to_fit:",parameters_to_fit)
	#print("params0:",params0)
	return parameters_to_fit , params0

def cov_jacobian(parameters_to_fit, cov_fm, tE, u0, rho, fs, finite_source: bool) -> np.ndarray:

	n = len(parameters_to_fit)
	J = np.eye(n)
	ln10 = np.log(10)
	for i, label in enumerate(parameters_to_fit):
		if label == 'log_tE':
			J[i,i] = 1.0 / (tE * ln10)
		elif not finite_source and label == 'log_u0':
			J[i,i] = 1.0 / (abs(u0) * ln10)
		elif finite_source and label == 'log_rho':
			J[i,i] = 1.0 / (rho * ln10)
		elif label == 'log_fs':
			J[i,i] = 1.0 / (fs * ln10)
		#else: leave J[i,i] = 1 for t0 and F0
	#print(J)
	cov_fm_log = J @ cov_fm @ J.T  
	#print(cov_fm_log)
	sig_cov_fm_log = np.sqrt(np.diag(cov_fm_log))
	#print(sig_cov_fm_log)
	return cov_fm_log, sig_cov_fm_log

def sig_fm(path):
	with open(path) as f:
		lines = f.read().splitlines()
	df0 = pd.DataFrame({'name': lines})
	df  = df0['name'].str.split(expand=True)

	fm_rows = df.iloc[54:].astype(float).to_numpy()

	sigma_t0   = fm_rows[1, 0]
	sigma_tE   = fm_rows[1, 1]
	sigma_u0   = fm_rows[1, 2]
	sigma_rho  = fm_rows[1, 3]
	sigma_F0   = fm_rows[1, 4]
	sigma_fs   = fm_rows[1, 5]

	sigmas_fm = [sigma_t0, sigma_tE, sigma_u0, sigma_rho, sigma_F0, sigma_fs]
	print("FM sigmas:", sigmas_fm)
	return sigmas_fm	
def read_fm(path, rho, u0, finite_source: bool):
	lines = open(path).read().splitlines()
        
	# Full BIJ matrix rows 1–10
	N = 6  

	bij_full = np.array(
		[ lines[i].split()[0:N] for i in range(1, 1+N) ],
		dtype=float
	)
	#print("bij_full:",bij_full)

	# Flip signs for negative u0
	if u0 < 0:
		bij_full[:, 2] *= -1
		bij_full[2, :] *= -1
		u0 = abs(u0)
	#print("bij_full:",bij_full)
	if not finite_source:
		idx = 3  # zero‐based index of 'rho'
		bij_reduced = np.delete(np.delete(bij_full, idx, axis=0),
						idx, axis=1)
	else:
		bij_reduced = bij_full

	# Invert reduced bij to get covariance
	cov_fm = np.linalg.inv(bij_reduced)
	sig_cov_fm = np.sqrt(np.diag(cov_fm))
	#print(cov_fm)
	#print(sig_cov_fm)
	return bij_reduced, u0, cov_fm, sig_cov_fm


def read_lc(path: str):
    data_df = pd.read_csv(path, delim_whitespace=True, header=None, comment='#')
    t = data_df.iloc[:,0].values
    F = data_df.iloc[:,1].values
    F_err = data_df.iloc[:,2].values
    
    fsm = pd.read_csv(path, delim_whitespace=True,header=None,comment=None,engine='python',nrows=4,index_col=False)
    
    return data_df, fsm, t, F, F_err


    
def compute_geometry(time, t0, tE, u0, alpha=0, a=0):
    # single-lens geometry for diagnostics
    slope = alpha * (np.pi/180)
    cosB, sinB = np.cos(slope), np.sin(slope)
    m1 = 1.0/(1 + 0)  # q=0 for single lens
    xcom = -m1 * a
    tt = (time - t0) / tE
    u = np.sqrt(u0**2 + tt**2)
    xs = u * cosB - tt * sinB + xcom
    ys = u * sinB + tt * cosB
    
    return xs, ys

