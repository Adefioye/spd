# exp_01_axis_aligned

Small axis-aligned calibration regime for testing SPD parameter recovery on the same 8-feature synthetic family across TMS and ResidMLP targets.

## Shared synthetic setup

- `num_features = 8`
- `hidden_dim = 8`
- dictionary initializer: `identity`
- firing distribution: uniform with `default_probability = 0.125`
- correlations: off
- hierarchy: off

## TMS dimensions

- input/output dimension `n_features = 8`
- bottleneck dimension `n_hidden = 4`
- SPD decomposes `linear1` and `linear2`
- SPD component budget: `C = 8` per decomposed layer

## ResidMLP dimensions

- input/output dimension `n_features = 8`
- embedding dimension `d_embed = 8`
- MLP width `d_mlp = 16`
- number of layers `n_layers = 2`
- embedding is fixed to identity
- SPD decomposes `layers.*.mlp_in` and `layers.*.mlp_out`
- SPD component budget: `C = 8` per decomposed layer

## Run commands

From the repo root:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_01_axis_aligned/tms.yaml
```

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_01_axis_aligned/resid_mlp.yaml
```

To train only the target model without running SPD:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_01_axis_aligned/tms.yaml \
  --stages target
```

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_01_axis_aligned/resid_mlp.yaml \
  --stages target
```

## Primary readout

Use the generated `parameter_recovery_analysis.json` to compare:

- `MMCS`
- `ML2R`
- total `faithfulness_mse`
- layerwise `faithfulness_mse`
