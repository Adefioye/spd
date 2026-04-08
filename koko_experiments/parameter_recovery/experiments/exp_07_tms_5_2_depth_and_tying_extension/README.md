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
- 10k SPD steps on CPU

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

- looks up the most recent saved target checkpoint for each base `exp_07` run under `SPD_OUT_DIR/parameter_recovery/results/`
- skips target retraining entirely
- keeps `wandb_project` from [exp_07_batch.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_batch.yaml)
- sets `wandb_run_name` to `<base_run_name>_steps40000`
- writes result bundles under `SPD_OUT_DIR/parameter_recovery/results/<base_run_name>_steps40000_<timestamp>/`

## Outputs

During execution, the runner materializes the exact configs used for each run under:

- `koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/materialized_configs/<run_name>/`

Run artifacts are written under `SPD_OUT_DIR`:

- target checkpoints: `SPD_OUT_DIR/train/<run_id>/` via the shared TMS trainer
- SPD runs: `SPD_OUT_DIR/spd/<run_id>/`
- rich analysis JSONs: `SPD_OUT_DIR/parameter_recovery/results/<run_name>_<timestamp>/`

Each result bundle contains:

- the batch manifest
- the exact materialized target config
- the exact materialized SPD config
- `<run_name>_parameter_recovery_analysis.json`

The most recent batch index is also written to:

- `materialized_configs/latest_run_index.json`

## 4-Layer Untied Loss Ablations

This folder also contains a focused SPD-only ablation batch for the existing
`exp_07_tms_5_2_4layer_untied` target. These runs reuse the saved 4-layer untied target checkpoint
and write outputs using the same `SPD_OUT_DIR/parameter_recovery/results/<run_name>_<timestamp>/`
layout as the main `exp_07` runner.

Runs:

- `exp_07_tms_5_2_4layer_untied_loss_ablation_baseline`
- `exp_07_tms_5_2_4layer_untied_loss_ablation_faithfulness`
- `exp_07_tms_5_2_4layer_untied_loss_ablation_faithfulness_unmasked`
- `exp_07_tms_5_2_4layer_untied_loss_ablation_faithfulness_impmin1e3`

Config and runner:

- batch config: [exp_07_4layer_untied_loss_ablations.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_4layer_untied_loss_ablations.yaml)
- runner: [run_exp_07_4layer_untied_loss_ablations.py](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_4layer_untied_loss_ablations.py)
- tmux launcher: [launch_4layer_untied_loss_ablations_tmux.sh](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_4layer_untied_loss_ablations_tmux.sh)

Run in tmux:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_4layer_untied_loss_ablations_tmux.sh
```

## 4-Layer Untied Faithfulness And P-Annealing Sweeps

This folder also contains a follow-up SPD-only sweep on the same reused
`exp_07_tms_5_2_4layer_untied` target checkpoint to test:

- `FaithfulnessLoss` coefficients `0.25`, `0.5`, `1.0`, `2.0`
- `ImportanceMinimalityLoss` p-annealing schedules `2.0 -> 1.0` and `2.0 -> 0.5`

Files:

- batch config: [exp_07_4layer_untied_faithfulness_panneal_sweeps.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_4layer_untied_faithfulness_panneal_sweeps.yaml)
- runner: [run_exp_07_4layer_untied_loss_ablations.py](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_4layer_untied_loss_ablations.py)
- tmux launcher: [launch_4layer_untied_faithfulness_panneal_sweeps_tmux.sh](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_4layer_untied_faithfulness_panneal_sweeps_tmux.sh)

Run in tmux:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_4layer_untied_faithfulness_panneal_sweeps_tmux.sh
```

## Untied 2L-6L Standard Setting

This folder also contains a reuse-target batch that applies the current best 4-layer untied loss
setting across all untied `TMS 5-2` depth variants from `2L` through `6L`:

- `FaithfulnessLoss coeff=1.0`
- `ImportanceMinimalityLoss coeff=3e-3`
- `ImportanceMinimalityLoss pnorm=2.0`
- `p_anneal_start_frac=0.0`
- `p_anneal_final_p=1.0`
- `p_anneal_end_frac=1.0`

Files:

- batch config: [exp_07_untied_standard_faithfulness_panneal.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_untied_standard_faithfulness_panneal.yaml)
- runner: [run_exp_07_4layer_untied_loss_ablations.py](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_4layer_untied_loss_ablations.py)
- tmux launcher: [launch_untied_standard_faithfulness_panneal_tmux.sh](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_untied_standard_faithfulness_panneal_tmux.sh)

Run in tmux:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_untied_standard_faithfulness_panneal_tmux.sh
```

## 6-Layer Untied Calibration Sweeps

This folder also contains a focused reuse-target sweep on the saved
`exp_07_tms_5_2_6layer_untied` target checkpoint to reduce the overgrowth seen from the
`FaithfulnessLoss=1.0` and `p_anneal 2.0 -> 1.0` setting.

Runs:

- `exp_07_tms_5_2_6layer_untied_f050_p2to15`
- `exp_07_tms_5_2_6layer_untied_f050_p2to125`
- `exp_07_tms_5_2_6layer_untied_f075_p2to15`
- `exp_07_tms_5_2_6layer_untied_f075_p2to125`

These test:

- `FaithfulnessLoss coeff=0.5` with `p_anneal 2.0 -> 1.5`
- `FaithfulnessLoss coeff=0.5` with `p_anneal 2.0 -> 1.25`
- `FaithfulnessLoss coeff=0.75` with `p_anneal 2.0 -> 1.5`
- `FaithfulnessLoss coeff=0.75` with `p_anneal 2.0 -> 1.25`

Files:

- batch config: [exp_07_6layer_untied_calibration_sweeps.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_6layer_untied_calibration_sweeps.yaml)
- runner: [run_exp_07_4layer_untied_loss_ablations.py](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_4layer_untied_loss_ablations.py)
- tmux launcher: [launch_6layer_untied_calibration_sweeps_tmux.sh](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_6layer_untied_calibration_sweeps_tmux.sh)

Run in tmux:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_6layer_untied_calibration_sweeps_tmux.sh
```

## 6-Layer Untied Faithfulness And PGD Sweeps

This folder also contains a focused reuse-target sweep on the saved
`exp_07_tms_5_2_6layer_untied` target checkpoint to test whether adding a small training-time
`PGDReconLoss` improves scale recovery in the hard `6L` untied setting.

Runs:

- `exp_07_tms_5_2_6layer_untied_f075_p2to15_pgd010`
- `exp_07_tms_5_2_6layer_untied_f075_p2to15_pgd025`
- `exp_07_tms_5_2_6layer_untied_f100_p2to15_pgd010`
- `exp_07_tms_5_2_6layer_untied_f100_p2to15_pgd025`

These use:

- `FaithfulnessLoss coeff in {0.75, 1.0}`
- `ImportanceMinimalityLoss p_anneal 2.0 -> 1.5`
- `PGDReconLoss coeff in {0.10, 0.25}`
- training PGD settings `init=random`, `step_size=1.0`, `n_steps=1`,
  `mask_scope=shared_across_batch`

Files:

- batch config: [exp_07_6layer_untied_faithfulness_pgd_sweeps.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_6layer_untied_faithfulness_pgd_sweeps.yaml)
- runner: [run_exp_07_4layer_untied_loss_ablations.py](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_4layer_untied_loss_ablations.py)
- tmux launcher: [launch_6layer_untied_faithfulness_pgd_sweeps_tmux.sh](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_6layer_untied_faithfulness_pgd_sweeps_tmux.sh)

Run in tmux:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_6layer_untied_faithfulness_pgd_sweeps_tmux.sh
```

## 6-Layer Untied Fixed-P Paper Follow-Up

This folder also contains a single reuse-target follow-up on the saved
`exp_07_tms_5_2_6layer_untied` target checkpoint that combines the best-performing PGD and
faithfulness settings from the previous `6L` PGD sweep with paper-style fixed `pnorm=1.0`.

Run:

- `exp_07_tms_5_2_6layer_untied_f100_p1fixed_pgd025`

This uses:

- `FaithfulnessLoss coeff=1.0`
- `ImportanceMinimalityLoss coeff=3e-3`
- `ImportanceMinimalityLoss pnorm=1.0`
- no `p`-annealing
- `PGDReconLoss coeff=0.25`

Files:

- batch config: [exp_07_6layer_untied_fixedp_paper_faithfulness_pgd.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/exp_07_6layer_untied_fixedp_paper_faithfulness_pgd.yaml)
- runner: [run_exp_07_4layer_untied_loss_ablations.py](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/run_exp_07_4layer_untied_loss_ablations.py)
- tmux launcher: [launch_6layer_untied_fixedp_paper_faithfulness_pgd_tmux.sh](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_6layer_untied_fixedp_paper_faithfulness_pgd_tmux.sh)

Run in tmux:

```bash
export SPD_OUT_DIR="$PWD/spd_out"
bash koko_experiments/parameter_recovery/experiments/exp_07_tms_5_2_depth_and_tying_extension/launch_6layer_untied_fixedp_paper_faithfulness_pgd_tmux.sh
```
