# Feature Recovery Experiments

This folder implements the executable version of `research_notes/feature_recovery.md`.

## What is here

- `synthetic.py`: thin `sae_lens` adapter for `FeatureDictionary`, `ActivationGenerator`, correlations, and hierarchy.
- `feature_datasets.py`: observed-activation datasets used by target-model training and SPD.
- `metrics.py`: common feature-recovery metrics for SAE and SPD, plus method-specific summaries.
- `results.py`: artifact saving/loading for SAE baselines and target-model bundles.
- `run_feature_recovery.py`: unified argparse CLI for SAE baseline, target training, SPD, and analysis.
- `aggregate_results.py`: aggregate `analysis.json` files into a JSONL table.
- `configs/axis_aligned_tms_sae_lens.yaml`: example axis-aligned TMS run with SAE baseline and SPD.
- `configs/hidden_activations_resid_mlp_sae_lens.yaml`: example hidden-activation ResidMLP run with SAE baseline.

## Stage layout

The current implementation collapses the earlier stage-0 to stage-3 plan into two core stages:

1. `axis_aligned`
   - synthetic family is driven by `sae_lens`
   - feature dictionary is identity
   - target models train directly on observed feature coordinates
   - use this as the calibration regime for both SAE and SPD
   - recommended experiment plan:
     1. Small clean calibration
        - `num_features = hidden_dim = 8`
        - SAE: `d_sae = 8`
        - TMS: `n_hidden = 4`
        - ResidMLP: `n_features = 8`, `d_embed = 8`, `fixed_embedding = identity`
        - data: uniform firing, `feature_probability` around `0.10` to `0.15`, no correlation, no hierarchy
        - purpose: establish the near-ceiling regime where both SAE and SPD should recover almost perfectly
     2. Medium matched-dimension baseline
        - `num_features = hidden_dim = 16`
        - SAE: `d_sae = 16`
        - TMS: `n_hidden = 8`
        - ResidMLP: `n_features = 16`, `d_embed = 16`, `fixed_embedding = identity`
        - data: start with uniform firing and `at_least_zero_active`, then add a Zipfian variant while keeping correlation and hierarchy off
        - purpose: check whether SAE and SPD stay aligned when the feature set is larger but still axis-aligned
     3. Large axis-aligned stress test
        - `num_features = hidden_dim = 32`
        - SAE: `d_sae = 32`
        - TMS: `n_hidden = 16`
        - ResidMLP: `n_features = 32`, `d_embed = 32`, `fixed_embedding = identity`
        - data: run three subconditions with the same dimensions
          - uniform firing only
          - Zipfian firing only
          - Zipfian plus mild correlations and shallow hierarchy
        - purpose: identify the first regime where SAE remains strong but SPD begins to fragment or merge even though the features are still axis-aligned
   - what to hold fixed for fair SAE vs SPD comparison:
     - keep the identity dictionary in all three experiments
     - set `d_sae` equal to the true number of features for the main matched-capacity run
     - use the same synthetic family, seeds, and evaluation sample count for SAE and target-model/SPD runs
     - for the cleanest comparison, start ResidMLP with `label_type = identity` and only add `act_plus_resid` after the identity-label baseline is stable
     - for SPD, use a matched component budget `C ~= num_features` first, then an overcomplete follow-up such as `C = 2 * num_features`
     - add one complication at a time: first firing distribution, then correlations, then hierarchy


2. `hidden_activations`
   - synthetic family is driven by `sae_lens`
   - target models train on dense hidden activations generated from known latent features
   - use dictionary geometry, firing distributions, correlations, and hierarchy to stress recovery

Both stages support the same synthetic knobs:

- firing probabilities: uniform or Zipfian
- dictionary geometry: identity, orthogonal, or random/non-orthogonal
- correlationality: random correlation matrices
- hierarchy: parent-child edges and mutually exclusive sibling groups

## Main entry point

Run the unified pipeline with:

```bash
python -m koko_experiments.feature_recovery.run_feature_recovery \
  koko_experiments/feature_recovery/configs/axis_aligned_tms_sae_lens.yaml
```

To run only a subset of stages:

```bash
python -m koko_experiments.feature_recovery.run_feature_recovery \
  koko_experiments/feature_recovery/configs/hidden_activations_resid_mlp_sae_lens.yaml \
  --stages sae,target
```

Available stages are:

- `sae`
- `target`
- `spd`
- `analyze`

If a config omits the `spd` block, the runner automatically skips `spd` and `analyze`.

## Evaluation

Common recovery metrics used for both SAE and SPD:

- mean matched cosine
- coverage@0.95
- coverage@0.99
- fragmentation
- merging

SAE-specific extras:

- explained variance
- true L0 / SAE L0
- dead latents
- shrinkage
- MCC
- uniqueness

SPD-specific extras:

- best recovered layer/direction role
- singleton-feature CI concentration summary

## Suggested order

1. Train the SAE baseline.
2. Train the target model on the same synthetic family.
3. Run SPD on the target bundle.
4. Compare SAE and SPD on the shared recovery metrics first, then inspect method-specific diagnostics.
