# README

This pipeline estimates microlensing parameter uncertainties for single lens events (including free-floating planets) and validates Fisher Matrix error approximations by directly comparing them to full MCMC posterior distributions.

## Usage

1. **Edit** `run_fit.py` to set:

   * `directory` pointing to your data files
   * `out_file` for the `.out` catalog
   * `outdir` where PDFs and summaries are saved
   * `obsgroup` list of observatory codes
   * `match` index for reference observatory

2. **Run** from the command line:

   ```bash
   python run_fit.py
   ```

3. For each event, the script:

   * Reads parameters from the ```gulls```  simulation outfile and decides whether to continue sampling in linear or logarithmic space
   * Reads Fisher matrix errors and decides point‐source vs. finite‐source model
   * Runs MCMC to sample the posterior
   * Produces a corner plot with FM vs. MCMC ellipses and a summary table of true parameter values vs. σ<sub>FM</sub> vs. σ<sub>MCMC</sub>
   * Generates a light‐curve fit and walker‐evolution diagnostic plots
   * Writes event summaries via `stats_utils`

## Customization

* **Sampling:** Tweak `n_walkers`, `n_steps`, `n_burn`, and `thin` in `initialize_emcee`.
