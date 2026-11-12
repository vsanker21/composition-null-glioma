# Composition null

Analysis scripts for the composition null applied to gene-program couplings in glioma.

The scripts, in the order of the locked analyses, are `run_null_standard.py`, `run_full_list.py`, `run_ivy_niche.py`, `run_visium_niche.py`, `run_mixture_bound.py`, `run_smooth_null.py`, `run_hallmark_census.py`, `run_gbmap_malignant.py`, and `run_within_donor.py`. `run_gates.py` supplies the shared scoring and regression functions.

Each script reads the cohort files named in its header and writes its own results directory. The thresholds in each script are fixed before the fit. No script is retuned after its output is read.

Licence: MIT.
