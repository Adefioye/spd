# Parameter Recovery Experiments

This folder contains SPD-only parameter-recovery experiments built on synthetic families generated with `sae_lens.synthetic`.

## What is here

- `synthetic.py`: synthetic feature dictionaries and activation generators, including firing-probability, correlation, and hierarchy controls.
- `feature_datasets.py`: observed-activation datasets used to train target models and run SPD.
- `metrics.py`: SPD parameter-recovery metrics centered on `MMCS`, `ML2R`, and paper-relevant loss summaries.
- `results.py`: target-bundle save/load helpers.
- `run_parameter_recovery.py`: unified CLI for target training, SPD, and analysis.
- `run_tms_single_with_analysis.py`: single-run wrapper for pretrained TMS SPD configs that saves a rich analysis bundle.
- `aggregate_results.py`: aggregate `parameter_recovery_analysis.json` files into a JSONL table.
- `experiments/`: concrete experiment folders with target YAMLs, dedicated SPD YAMLs, and per-experiment notes.

## Experimental stages

Two synthetic stages are supported:

1. `axis_aligned`
   - identity feature dictionary
   - target models train directly on observed feature coordinates
   - use this to calibrate how well SPD can recover clean ground-truth features from parameters

2. `hidden_activations`
   - non-identity feature dictionary
   - target models train on dense hidden activations generated from known sparse latent features
   - use this to stress SPD under superposition, non-orthogonality, correlations, and hierarchy

Both stages support the same synthetic dials:

- firing probabilities: uniform, manual, or Zipfian
- geometry: identity, orthogonal, or random dictionaries
- correlationality: random correlation matrices
- hierarchy: parent-child edges and mutually exclusive groups

## Primary metrics

Feature-recovery analysis is SPD-only and focuses on:

- `MMCS`: mean max cosine similarity between each true feature and its best-matching recovered direction
- `ML2R`: mean L2 ratio between matched recovered directions and true feature norms

The analysis output also includes:

- layerwise `faithfulness_mse`
- total `faithfulness_mse` across analyzed layers
- paper-relevant SPD losses from `metrics.jsonl`
- CI concentration summaries for singleton-feature probes

## Main entry point

Run a concrete experiment YAML from `experiments/`, for example:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_01_axis_aligned/tms.yaml
```

Available stages are:

- `target`
- `spd`
- `analyze`

Examples:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_01_axis_aligned/tms.yaml \
  --stages target
```

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_parameter_recovery \
  koko_experiments/parameter_recovery/experiments/exp_01_axis_aligned/resid_mlp.yaml
```

For a pretrained TMS SPD YAML that should save directly into the parameter-recovery results folder, use:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_tms_single_with_analysis \
  koko_experiments/parameter_recovery/experiments/exp_04_tms_5_2_paper_with_faithfulness_train/tms_5-2_cpu_paper_with_faithfulness_train.yaml
```

Target training can now use either:

- a streamed synthetic distribution when `target.train_num_samples` is omitted
- a finite materialized dataset when `target.train_num_samples` is set

Use `target.eval_num_samples` the same way for evaluation. This only changes target-model training and eval. The SPD stage still uses the synthetic distribution defined by the SPD config.

## Output layout

By default, outputs are written under `SPD_OUT_DIR`:

- target bundles: `SPD_OUT_DIR/parameter_recovery/targets/<run_name>_<timestamp>/`
- analysis bundles: `SPD_OUT_DIR/parameter_recovery/results/<run_name>_<timestamp>/`
- SPD runs: `SPD_OUT_DIR/spd/<run_id>/`

The SPD run still contains:

- `SPD_OUT_DIR/spd/<run_id>/parameter_recovery_analysis.json`

The cleaner summary artifact is also written to:

- `SPD_OUT_DIR/parameter_recovery/results/<run_name>_<timestamp>/<run_name>_parameter_recovery_analysis.json`

This file contains:

- total `MMCS`
- total `ML2R`
- total `faithfulness_mse`
- paper-relevant loss summaries
- layerwise recovery breakdown

Single-run wrappers write only this rich analysis JSON into the results bundle. Separate summary files are produced only by replicate-oriented scripts such as `run_tms_paper_replicates.py`.

## Suggested order

1. Train the target model on one synthetic family.
2. Run SPD on that target bundle.
3. Compare `MMCS` and `ML2R` first.
4. Use layerwise `faithfulness_mse`, loss summaries, and CI summaries to diagnose failure modes.
