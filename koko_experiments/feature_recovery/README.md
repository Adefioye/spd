# Feature Recovery Experiments

This folder implements the executable version of `research_notes/feature_recovery.md`.

## What is here

- `synthetic.py`: known ground-truth feature dictionaries, Zipfian probabilities, correlated activations, hierarchy.
- `feature_datasets.py`: batch generators for hidden activations and ResidMLP target-model training.
- `autoencoder.py`: a lightweight sparse autoencoder baseline for stage-0 calibration.
- `metrics.py`: dictionary-recovery and SPD-direction metrics.
- `train_sae_baseline.py`: stage-0 synthetic recovery baseline.
- `train_resid_mlp_target.py`: train ResidMLP targets with identity/random/known-dictionary embeddings.
- `run_resid_mlp_spd.py`: run SPD on local feature-recovery target bundles.
- `run_tms_feature_recovery.py`: thin wrapper around the repo's TMS train + SPD path.
- `analyze_spd_run.py`: compare SPD components to the known feature dictionary.
- `aggregate_results.py`: aggregate `analysis.json` files into a JSONL table.

## Suggested order

1. Train the SAE baseline.
2. Train a dictionary-initialized ResidMLP target.
3. Run SPD on the target bundle.
4. Analyze the SPD run against the true feature dictionary.
5. Explore the resulting artifacts in the notebooks under `koko_notebooks/`.
