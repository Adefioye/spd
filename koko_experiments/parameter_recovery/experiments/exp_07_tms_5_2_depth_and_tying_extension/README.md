# exp_07_tms_5_2_depth_and_tying_extension

This experiment reproduces the paper-style `TMS_{5-2}` sparse-feature target training setup and
extends it to deeper targets and untied `linear1`/`linear2` weights.

The ten runs are:

- `exp_07_tms_5_2_2layer_tied`
- `exp_07_tms_5_2_2layer_untied`
- `exp_07_tms_5_2_3layer_tied`
- `exp_07_tms_5_2_3layer_untied`
- `exp_07_tms_5_2_4layer_tied`
- `exp_07_tms_5_2_4layer_untied`
- `exp_07_tms_5_2_5layer_tied`
- `exp_07_tms_5_2_5layer_untied`
- `exp_07_tms_5_2_6layer_tied`
- `exp_07_tms_5_2_6layer_untied`

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
- 10k target-training steps on CPU
- AdamW with constant LR `5e-3` and `weight_decay: 0.01`

Shared SPD settings:

- paper-style TMS 5-2 hyperparameters
- `C=20` per decomposed layer
- CI MLP hidden width `16`
- `ImportanceMinimalityLoss coeff=3e-3, pnorm=1`
- `StochasticReconLoss coeff=1`
- `StochasticReconLayerwiseLoss coeff=1`
- 40k SPD steps on CPU

All runs log to the single W&B project declared in
[exp_07_batch.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_batch.yaml).

## Run In tmux

From the repo root:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_tmux.sh
```

This launcher:

- exports `SPD_OUT_DIR="$PWD/spd_out"`
- forces CPU execution with `CUDA_VISIBLE_DEVICES=''`
- starts one tmux session that runs all ten experiments sequentially
- writes a central stdout/stderr log into `logs/`

To attach:

```bash
tmux attach -t exp07_tms_depth
```

## Run A Subset

You can run selected experiments directly:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
export CUDA_VISIBLE_DEVICES=""
source .venv/bin/activate
python koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_batch.py \
  koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_batch.yaml \
  --run-names exp_07_tms_5_2_2layer_tied,exp_07_tms_5_2_3layer_tied
```

## Reuse Saved Targets For Longer SPD Runs

You can rerun SPD against the already-saved `exp_07` target checkpoints without retraining targets.
This is useful for longer decompositions, such as a 40k-step sweep that keeps the same W&B project
while giving each SPD run and result bundle a distinct suffix.

From the repo root:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
export CUDA_VISIBLE_DEVICES=""
source .venv/bin/activate
python koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_batch.py \
  koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_batch.yaml \
  --reuse-existing-targets \
  --spd-steps 40000 \
  --run-name-suffix steps40000
```

Behavior:

- looks up the most recent saved analysis JSON for each base `exp_07` run under `SPD_OUT_DIR/parameter_recovery/results/`
- skips target retraining entirely
- keeps `wandb_project` from [exp_07_batch.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_batch.yaml)
- sets `wandb_run_name` to `<base_run_name>_steps40000`
- writes analysis JSONs under `SPD_OUT_DIR/parameter_recovery/results/<base_run_name>_steps40000_<timestamp>_parameter_recovery_analysis.json`

## Outputs

During execution, the runner materializes the exact configs used for each run under:

- `koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/materialized_configs/<run_name>/`

Run artifacts are written under `SPD_OUT_DIR`:

- target checkpoints: `SPD_OUT_DIR/train/<run_id>/` via the shared TMS trainer
- SPD runs: `SPD_OUT_DIR/spd/<run_id>/`
- rich analysis JSONs: `SPD_OUT_DIR/parameter_recovery/results/<run_name>_<timestamp>_parameter_recovery_analysis.json`

Each saved result is a single JSON file:

- `<run_name>_<timestamp>_parameter_recovery_analysis.json`

The exact batch, target, and SPD configs are still materialized under:

- `koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/materialized_configs/<run_name>/`

The most recent batch index is also written to:

- `materialized_configs/latest_run_index.json`

## Delta-Component Shrinkage Ablation

This folder also includes a dedicated 5-layer and 6-layer untied ablation for investigating
shrinkage under `use_delta_component` and `FaithfulnessLoss`.

The six runs are:

- `exp_07_tms_5_2_5layer_untied_deltaon_nofaith_steps40000`
- `exp_07_tms_5_2_5layer_untied_deltaoff_faith_steps40000`
- `exp_07_tms_5_2_5layer_untied_deltaon_faith_steps40000`
- `exp_07_tms_5_2_6layer_untied_deltaon_nofaith_steps40000`
- `exp_07_tms_5_2_6layer_untied_deltaoff_faith_steps40000`
- `exp_07_tms_5_2_6layer_untied_deltaon_faith_steps40000`

These runs keep the paper-style TMS 5-2 SPD settings for the non-ablated terms:

- `ImportanceMinimalityLoss coeff=3e-3, pnorm=1`
- `StochasticReconLoss coeff=1`
- `StochasticReconLayerwiseLoss coeff=1`
- `C=20` per decomposed layer
- CI MLP hidden width `16`
- cosine LR schedule from `1e-3`
- `40_000` SPD steps

Variant definitions:

- `deltaon_nofaith`: `use_delta_component: true` with no `FaithfulnessLoss`
- `deltaoff_faith`: `use_delta_component: false` with `FaithfulnessLoss coeff=1`
- `deltaon_faith`: `use_delta_component: true` with `FaithfulnessLoss coeff=1`

Launch the full ablation batch in `tmux` from the repo root:

```bash
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_delta_component_shrinkage_tmux.sh
```

This launcher:

- exports `SPD_OUT_DIR="$PWD/spd_out"`
- forces CPU execution with `CUDA_VISIBLE_DEVICES=''`
- runs the six experiments sequentially in one detached `tmux` session
- writes a dedicated log file into `logs/`
