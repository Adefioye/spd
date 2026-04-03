# exp_05_tms_5_2_hyperparameter_grid_search

Focused grid search over the five SPD paper hyperparameters for the Goodfire pretrained
TMS 5-2 target used in `exp_04`, while keeping `FaithfulnessLoss` in the main training objective
and reducing SPD training to `5_000` steps per run.

Swept hyperparameters:

- `n_mask_samples`
- `StochasticReconLoss.coeff`
- `StochasticReconLayerwiseLoss.coeff`
- `ImportanceMinimalityLoss.coeff`
- `ImportanceMinimalityLoss.pnorm`

The base config is [tms_5-2_paper_with_faithfulness_grid_search.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_05_tms_5_2_hyperparameter_grid_search/tms_5-2_paper_with_faithfulness_grid_search.yaml), and the explicit grid values are in [sweep_plan.json](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_05_tms_5_2_hyperparameter_grid_search/sweep_plan.json).

This sweep is intentionally narrower than a brute-force `5^5` search. The values are chosen to stay
close to the paper:

- `ImportanceMinimalityLoss.coeff` (`beta_3`) spans the TMS values reported in the paper, from
  `1e-4` for `TMS_40-10` up to `3e-3` for `TMS_5-2`, with denser coverage toward the `TMS_5-2`
  end: `1e-4, 3e-4, 1e-3, 2e-3, 3e-3`
- `ImportanceMinimalityLoss.pnorm` uses only `1.0` and `2.0`, the two TMS values actually used in
  the paper
- `StochasticReconLoss.coeff` and `StochasticReconLayerwiseLoss.coeff` are centered on the paper
  default `1.0`, with only one octave below and above: `0.5, 1.0, 2.0`
- `n_mask_samples` uses `1` and `4`; the paper reports `S=1` as sufficient, so higher values are
  treated only as a targeted variance-reduction check rather than a broad sweep

That gives `2 x 3 x 3 x 5 x 2 = 180` total runs, which is substantially smaller while remaining
grounded in the paper's stated TMS settings and hyperparameter heuristics.

Run from the repo root:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_tms_hyperparameter_grid_search \
  koko_experiments/parameter_recovery/experiments/exp_05_tms_5_2_hyperparameter_grid_search
```

This command does all of the following:

- materializes one labeled SPD config per hyperparameter combination
- runs all `180` SPD sweeps
- resumes the latest partial result folder by default if a previous `exp_05` sweep was interrupted
- computes only aggregate recovery metrics per sweep: all-layer `MMCS` and all-layer `ML2R`
- saves one compact JSON per sweep under `per_run_metrics/`
- writes a full `mmcs_ml2r_sweep_summary.json` ranked by `0.5 * (MMCS + ML2R)`
- writes `top_5_hyperparameter_combinations.json`

W&B naming:

- `wandb_run_name` includes the full hyperparameter label for the sweep combination
- `wandb_project` is also stamped with the experiment identifier and hyperparameter label for the
  run, so interrupted combinations are easy to identify unambiguously in W&B

Where to find the final result:

- all sweep outputs: `SPD_OUT_DIR/parameter_recovery/results/exp_05_tms_5_2_hyperparameter_grid_search_<timestamp>/`
- per-sweep compact metrics: `.../per_run_metrics/*_sweep_metrics.json`
- full ranked table: `.../mmcs_ml2r_sweep_summary.json`
- best five hyperparameter combinations: `.../top_5_hyperparameter_combinations.json`

To rerun only the ranking step on an existing result folder:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.summarize_tms_hyperparameter_grid_search \
  /path/to/SPD_OUT_DIR/parameter_recovery/results/exp_05_tms_5_2_hyperparameter_grid_search_<timestamp>
```

Resume behavior:

- default behavior: resume the latest incomplete `exp_05` result folder automatically
- resume a specific partial result folder:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_tms_hyperparameter_grid_search \
  koko_experiments/parameter_recovery/experiments/exp_05_tms_5_2_hyperparameter_grid_search \
  --result-root /path/to/SPD_OUT_DIR/parameter_recovery/results/exp_05_tms_5_2_hyperparameter_grid_search_<timestamp>
```

- force a brand-new sweep instead of resuming:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_tms_hyperparameter_grid_search \
  koko_experiments/parameter_recovery/experiments/exp_05_tms_5_2_hyperparameter_grid_search \
  --fresh
```
