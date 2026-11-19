# Composition null

Analysis scripts for the composition null applied to gene-program couplings in glioma.

The scripts for the prespecified analyses are `run_null_standard.py`, `run_full_list.py`, `run_ivy_niche.py`, `run_visium_niche.py`, `run_mixture_bound.py`, `run_smooth_null.py`, `run_hallmark_census.py`, `run_gbmap_malignant.py`, and `run_within_donor.py`. `run_gates.py` supplies the shared scoring and regression functions.

The marker-mixture and reliability scripts are `run_raw_marker.py`, `run_saturating_marker.py`, `run_neuron_marker.py`, `run_neuronal_saturating.py`, `run_joint_marker.py`, and `run_spot_reliability.py`. `run_raw_marker.py` imports `run_depth_and_marker.py`. The donor-level model with detected genes is `run_donor_depth.py`. Each of these was written before its own output was read.

Each script reads the cohort files named in its header and writes its own results directory. The thresholds in each script are fixed before the fit. No script is retuned after its output is read.

Licence: MIT.
