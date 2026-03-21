# exp_00_axis_aligned_tms_5_2

Paper-style `TMS_5-2` axis-aligned parameter-recovery experiment.

This is the closest current match to the paper setup using the `sae_lens.synthetic` backend.

Important caveat:

- the original paper-style TMS data uses sparse features with active values sampled uniformly from `[0, 1]`
- the current `sae_lens` wrapper samples active values as `ReLU(mean + Gaussian noise)`
- so this experiment is a reproduction proxy, not an exact synthetic-data match

## Dimensions

- `num_features = 5`
- `hidden_dim = 5`
- TMS target: `n_features = 5`, `n_hidden = 2`
- TMS weights: `tied_weights = true`
- SPD component budget: `C = 20` for `linear1` and `linear2`

## Synthetic setup

- axis-aligned identity dictionary
- firing probability `0.05`
- no correlations
- no hierarchy
- finite target dataset:
  - `train_num_samples = 1,048,576`
  - `eval_num_samples = 65,536`
- activation magnitude proxy for paper data:
  - `mean_firing_magnitudes = 0.5`
  - `std_firing_magnitudes = 0.288675`

## Run

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_00_axis_aligned_tms_5_2/tms.yaml
```

## What to compare

Use the generated `parameter_recovery_analysis.json` to inspect:

- total `MMCS`
- total `ML2R`
- per-layer `MMCS`
- per-layer `ML2R`
- total and layerwise `faithfulness_mse`
- SPD losses from `metrics.jsonl`
