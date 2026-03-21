# exp_01_axis_aligned

Small axis-aligned calibration regime for comparing SAE and SPD on the same 8-feature synthetic family.

## Shared synthetic setup

- `num_features = 8`
- `hidden_dim = 8`
- dictionary initializer: `identity`
- firing distribution: uniform with `default_probability = 0.125`
- correlations: off
- hierarchy: off

## SAE dimensions

- input dimension `d_in = 8`
- latent dimension `d_sae = 8`

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
python -m koko_experiments.feature_recovery.run_feature_recovery \
  koko_experiments/feature_recovery/experiments/exp_01_axis_aligned/tms.yaml
```

```bash
source .venv/bin/activate
python -m koko_experiments.feature_recovery.run_feature_recovery \
  koko_experiments/feature_recovery/experiments/exp_01_axis_aligned/resid_mlp.yaml
```

To run only the SAE baseline and target training without SPD:

```bash
source .venv/bin/activate
python -m koko_experiments.feature_recovery.run_feature_recovery \
  koko_experiments/feature_recovery/experiments/exp_01_axis_aligned/tms.yaml \
  --stages sae,target
```

```bash
source .venv/bin/activate
python -m koko_experiments.feature_recovery.run_feature_recovery \
  koko_experiments/feature_recovery/experiments/exp_01_axis_aligned/resid_mlp.yaml \
  --stages sae,target
```
