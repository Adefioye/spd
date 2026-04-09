# exp_08_tms_40_10_depth_untied_extension

This experiment extends the `exp_07` depth study to `TMS_{40-10}` using only untied targets across
depths `2L` through `6L`.

The five runs are:

- `exp_08_tms_40_10_2layer_untied_steps40000`
- `exp_08_tms_40_10_3layer_untied_steps40000`
- `exp_08_tms_40_10_4layer_untied_steps40000`
- `exp_08_tms_40_10_5layer_untied_steps40000`
- `exp_08_tms_40_10_6layer_untied_steps40000`

Tied counterparts are also supported in the same folder via `exp_08_tied_batch.yaml`:

- `exp_08_tms_40_10_2layer_tied_steps40000`
- `exp_08_tms_40_10_3layer_tied_steps40000`
- `exp_08_tms_40_10_4layer_tied_steps40000`
- `exp_08_tms_40_10_5layer_tied_steps40000`
- `exp_08_tms_40_10_6layer_tied_steps40000`

Layer naming in this folder:

- `2layer`: standard `linear1 -> linear2` TMS, so `n_hidden_layers=0`
- `3layer`: `linear1 -> hidden_layers.0 -> linear2`, so `n_hidden_layers=1`
- `4layer`: `linear1 -> hidden_layers.0 -> hidden_layers.1 -> linear2`, so `n_hidden_layers=2`
- `5layer`: `linear1 -> hidden_layers.0 -> hidden_layers.1 -> hidden_layers.2 -> linear2`, so `n_hidden_layers=3`
- `6layer`: `linear1 -> hidden_layers.0 -> hidden_layers.1 -> hidden_layers.2 -> hidden_layers.3 -> linear2`, so `n_hidden_layers=4`

Shared target-training settings:

- sparse-feature dataset with `feature_probability: 0.05`
- `data_generation_type: at_least_zero_active`
- input values sampled from `[0, 1]`
- `10k` target-training steps
- untied `linear1`/`linear2`
- `n_features=40`, `n_hidden=10`
- AdamW with constant LR `5e-3` and `weight_decay: 0.01`

Shared SPD settings:

- `TMS-40-10` decomposition hyperparameters
- `C=200` per decomposed layer
- `ImportanceMinimalityLoss coeff=1e-4, pnorm=2`
- `StochasticReconLoss coeff=1`
- `StochasticReconLayerwiseLoss coeff=1`
- `40k` SPD steps

All runs log to the single W&B project declared in
[exp_08_batch.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/exp_08_batch.yaml).

## Run In tmux

From the repo root on a GPU machine:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/launch_tmux.sh
```

This launcher:

- exports `SPD_OUT_DIR="$PWD/spd_out"`
- starts one tmux session
- defaults to `--device cuda`
- writes a central stdout/stderr log into `logs/`

To run the tied 2L-6L batch in the same folder:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/launch_tmux.sh \
  exp08_tms_40_10_tied \
  cuda \
  exp_08_tied_batch.yaml
```

To attach:

```bash
tmux attach -t exp08_tms_40_10_untied
```

To force CPU instead:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/launch_tmux.sh exp08_tms_40_10_untied cpu
```

## Run A Subset

You can run selected experiments directly:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
source .venv/bin/activate
python koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/run_exp_08_batch.py \
  koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/exp_08_batch.yaml \
  --device cuda \
  --run-names exp_08_tms_40_10_2layer_untied_steps40000,exp_08_tms_40_10_3layer_untied_steps40000
```

For tied runs, swap to `exp_08_tied_batch.yaml` and tied run names.

## Reuse Saved Targets

The runner also supports reusing previously saved target checkpoints:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
source .venv/bin/activate
python koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/run_exp_08_batch.py \
  koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/exp_08_batch.yaml \
  --device cuda \
  --reuse-existing-targets
```

## Outputs

During execution, the runner materializes the exact configs used for each run under:

- `koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/materialized_configs/<run_name>/`

Run artifacts are written under `SPD_OUT_DIR`:

- target checkpoints: `SPD_OUT_DIR/train/<run_id>/`
- SPD runs: `SPD_OUT_DIR/spd/<run_id>/`
- rich analysis JSONs: `SPD_OUT_DIR/parameter_recovery/results/<run_name>_<timestamp>_parameter_recovery_analysis.json`

Each saved result is a single JSON file:

- `<run_name>_<timestamp>_parameter_recovery_analysis.json`

The exact batch, target, and SPD configs are still materialized under:

- `koko_experiments/parameter_recovery/experiments/exp_08_tms_40_10_depth_untied_extension/materialized_configs/<run_name>/`
