# exp_06_tms_5_2_top_hparams_single_run

Single paper-style TMS 5-2 SPD run using the same pretrained target model and training
schedule as `exp_04`, but with the selected `exp_05` hyperparameters:

- `n_mask_samples = 1`
- `StochasticReconLoss.coeff = 0.5`
- `StochasticReconLayerwiseLoss.coeff = 0.5`
- `ImportanceMinimalityLoss.coeff = 3e-3`
- `ImportanceMinimalityLoss.pnorm = 2.0`

Run from the repo root:

```bash
source .venv/bin/activate
export SPD_OUT_DIR="$PWD/spd_out"
python -m koko_experiments.parameter_recovery.run_tms_single_with_analysis \
  koko_experiments/parameter_recovery/experiments/exp_06_tms_5_2_top_hparams_single_run/tms_5-2_cpu_top_hparams_train.yaml
```

This command:

- runs the SPD training job
- writes `parameter_recovery_analysis.json` into the SPD run directory
- writes the analysis JSON into `SPD_OUT_DIR/parameter_recovery/results/...`
- copies the exact config used into the result bundle
