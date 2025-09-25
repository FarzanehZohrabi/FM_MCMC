import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

def A(F, fs):
    return (F - (1 - fs)) / fs

def mag(F, ms, fs):
    m0 = ms + 2.5 * np.log10(fs)
    return m0 - 2.5 * np.log10(F)

def magerr(F, e, ms, fs):
    return (2.5 / np.log(10)) * e / F
    


def plot_corner_with_ellipses(
    samples: np.ndarray,
    parameters_to_fit: list,
    cov_mcmc: np.ndarray,
    cov_fm: np.ndarray,
    params_gulls: list,
    sigma_fm: np.ndarray,
    sigma_mcmc: np.ndarray,
    finite_source: bool,
    output_pdf: PdfPages,
    title: str = "FFP Microlensing Parameter Uncertainties: FM Estimates vs MCMC Posterior"
):
    n = len(parameters_to_fit)
    # Choose labels only for the keys you actually have
    base_map = {
        "t0":   r"$t_0$",
        "log_t0": r"$\log t_0$",
        "tE":   r"$t_E$",
        "log_tE": r"$\log t_E$",
        "u0":   r"$u_0$",
        "log_u0": r"$\log u_0$",
        "rho":  r"$\rho$",
        "log_rho": r"$\log \rho$",
        "F0" : r"$F_0$",
        "fs":   r"$f_s$",
        "log_fs": r"$\log f_s$"
    }

    labels = [base_map[p] for p in parameters_to_fit]
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['axes.formatter.limits'] = [-2, 2]
    mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman','Palatino','Georgia'],
    'font.size': 16,
    'axes.titlesize': 30,
    'axes.labelsize': 18,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#666666',
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,})

    # Set up figure with constrained layout
    fig = plt.figure(figsize=(3*n + 4, 3*n), constrained_layout=False)
    gs = fig.add_gridspec(n, n + 1, width_ratios=[1]*n + [0.70],wspace=0.02, hspace=0.02)

    # Corner on left nxn subgrid
    corner_axs = np.array([
        [fig.add_subplot(gs[i, j]) for j in range(n)]
        for i in range(n)
    ])
    corner.corner(
        samples,
        labels=labels,
        figsize=(1.5 * n, 1.5 * n),
        show_titles=True,
        title_fmt=".3f",
        fig=fig,
        axes=corner_axs,
        label_kwargs={"fontsize":20, "labelpad":2},
        title_kwargs={"fontsize":20, "pad":2},
        quantiles=[0.158, 0.5, 0.842],
        levels=[0.158, 0.842],   # just 1σ filled
        plot_contours=False,
        fill_contours=True,
        contour_kwargs={"colors":["tab:blue","tab:red"], "alpha":0.7, "linewidths":[1.5,1.5]},
        hist_kwargs={"density":False, "color":"gray", "alpha":0.9},
        plot_datapoints=True,
        space=0.03,
        use_math_text=True
    )

    # Overplot Fisher vs MCMC ellipses
    med = np.median(samples, axis=0)
    θ = np.linspace(0, 2*np.pi, 200)
    for i in range(1, n):
        for j in range(i):
            ax = corner_axs[i,j]
            # Fisher (red dashed)
            Cf = cov_fm[np.ix_([j,i],[j,i])]
            if np.any(Cf):
                vals, vecs = np.linalg.eigh(Cf)
                order = np.argsort(vals)[::-1]
                vals, vecs = vals[order], vecs[:,order]
                ang = np.arctan2(vecs[1,0], vecs[0,0])
                a,b = np.sqrt(2.28*vals)
                R = np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang),  np.cos(ang)]])
                ell = np.column_stack([a*np.cos(θ), b*np.sin(θ)]).dot(R.T)
                ax.plot(ell[:,0]+med[j], ell[:,1]+med[i],
                        c="tab:red", ls="--", lw=2)
            # MCMC (blue solid)
            Cc = cov_mcmc[np.ix_([j,i],[j,i])]
            vals, vecs = np.linalg.eigh(Cc)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:,order]
            ang = np.arctan2(vecs[1,0], vecs[0,0])
            a,b = np.sqrt(2.28*vals)
            R = np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang),  np.cos(ang)]])
            ell = np.column_stack([a*np.cos(θ), b*np.sin(θ)]).dot(R.T)
            ax.plot(ell[:,0]+med[j], ell[:,1]+med[i],
                    c="tab:blue", ls="-", lw=2)

    fig.legend(
        handles=[
            Line2D([0],[0], c="tab:blue", lw=2, label="MCMC"),
            Line2D([0],[0], c="tab:red",  lw=2, ls="--", label="Fisher")
        ],
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=20,
        bbox_to_anchor=(0.95, 0.63)
    )


    tab_ax = fig.add_subplot(gs[:, -1])
    tab_ax.axis("off")
    tab_ax.set_position([0.7, 0.10, 0.15, 0.78]) 
    table_data = [
        [base_map[p],
         f"{params_gulls[idx]:.3g}",
         f"{sigma_fm[idx]:.3g}",
         f"{sigma_mcmc[idx]:.3g}"]
        for idx, p in enumerate(parameters_to_fit)
    ]
    cols = ["Parameter", "Value", r"$\sigma_{\rm FM}$", r"$\sigma_{\rm MCMC}$"]
    tbl = tab_ax.table(
        cellText=table_data,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.35, 0.35, 0.35],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(20)
    tbl.scale(1.5, 2.5)



    #fig.suptitle(title, fontsize=30, x=0.63,y=0.95)
    output_pdf.savefig(fig)
    plt.close(fig)



def plot_lc(t0,tE,data_df,
            fsm,
            match: int = 0,
            nobs: int = 3,
            displayobs: list = None,
            labels: list = None,
            obsgroup: list = None,
            models: list = None,
            output_pdf: PdfPages = None):

    from cycler import cycler
    plt.rc('axes', prop_cycle=cycler('color', ['y','c','m']))
    fig, ax = plt.subplots(figsize=(15,10))
    fs0 = float(fsm.iloc[0, match+1])
    ms0 = float(fsm.iloc[-1, match+1])
    m0 = ms0 + 2.5*np.log10(fs0)
    lc_data = data_df.iloc[8:].to_numpy(dtype=float)
    if displayobs is None:
        displayobs = list(range(nobs))
    for ii, obs in enumerate(displayobs):
        d = lc_data[lc_data[:,5] == obs]
        fs = float(fsm.iloc[0, obs+1])
        mu = A(d[:,1], fs)
        mi = mag(fs0*mu + 1 - fs0, ms0, fs0)
        sigmi = magerr(fs0*mu + 1 - fs0, d[:,2], ms0, fs0)
        ax.errorbar(d[:,0], mi, yerr=sigmi, fmt='o', ms=8, label=obsgroup[ii])
    if models is not None:
        for model_t in models:
            ax.plot(lc_data[lc_data[:,5]==match,0], model_t, alpha=0.7)
    d_match = lc_data[lc_data[:,5] == match]
    fs = float(fsm.iloc[0, match+1])
    mu_true = A(d_match[:,3], fs)
    mu_fit  = A(d_match[:,7], fs)
    mitrue  = mag(fs0*mu_true + 1 - fs0, ms0, fs0)
    mifit   = mag(fs0*mu_fit  + 1 - fs0, ms0, fs0)
    ax.plot(d_match[:,0], mitrue, '-', color='k', alpha=0.5, zorder=10)
    #ax.plot(d_match[:,0], mifit, '--', color='r', zorder=11)
    ax.set(xlabel='Time (days)', ylabel='Magnitude')
    ax.set(xlim=(t0-1,t0+1))
    ax.tick_params(labelsize=20)
    ax.invert_yaxis()
    ax.legend(fontsize=20)
    if output_pdf is not None:
        output_pdf.savefig(fig)
        plt.close(fig)


def plot_parameter_evolution(sampler,
							parameters_to_fit: list,
							output_pdf: PdfPages):
	"""
	Plot the MCMC walker evolution for each parameter.
	"""
	chain = sampler.get_chain()     # shape = (n_walkers, n_steps, n_dim)
	n_dim = chain.shape[-1]
	fig, axes = plt.subplots(n_dim, 1, figsize=(8,2*n_dim), sharex=True)
	for i in range(n_dim):
		axes[i].plot(chain[:,:,i], alpha=0.3)
		axes[i].set_ylabel(parameters_to_fit[i])
		axes[i].yaxis.set_label_coords(-0.1, 0.5)
	axes[-1].set_xlabel("step number")
	output_pdf.savefig(fig)
	plt.close(fig)

