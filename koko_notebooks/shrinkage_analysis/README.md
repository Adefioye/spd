# Shrinkage Analysis Notebooks

This folder contains a checkpoint-oriented notebook suite for `exp_07_tms_5_2_*` runs.

The suite uses the following reframing:

- `ML2R` is the canonical paper-style shrinkage metric
- raw weight ratios diagnose raw under-scaling
- masked ratios diagnose CI attenuation
- output and Jacobian ratios diagnose functional contraction

Recommended order:

1. `00_run_inventory_and_manifest.ipynb`
2. `01_raw_and_delta_analysis.ipynb`
3. `02_ci_and_mask_analysis.ipynb`
4. `03_functional_and_jacobian_analysis.ipynb`
5. `04_matching_and_feature_recovery.ipynb`
6. `05_summary_plots_and_interpretation.ipynb`

Shared helpers live in [shrinkage_analysis_common.py](/workspace/spd/koko_notebooks/shrinkage_analysis/shrinkage_analysis_common.py).
Publication plot helpers live in [publication_plots.py](/workspace/spd/koko_notebooks/shrinkage_analysis/publication_plots.py).

The notebooks are designed to work incrementally:

- inventory runs and checkpoints first
- save intermediate CSVs under `koko_notebooks/shrinkage_analysis/outputs/`
- save publication-grade figures under `koko_notebooks/shrinkage_analysis/outputs/plots/`
- write per-notebook plot manifests as JSON alongside those figures
- combine the stage CSVs only in the final summary notebook

That keeps per-notebook runtime and memory manageable when sweeping many checkpoints.
