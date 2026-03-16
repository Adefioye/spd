# SPD Metrics Glossary (Run `s-c6eda15c`)

This glossary maps the exact metrics in:
- `spd_out/spd/s-c6eda15c/metrics.jsonl`
- `spd_out/tms_alignment/s-c6eda15c/alignment_report_latest.json`

to what they mean and where they are implemented.

## Run Context

- Run dir: `spd_out/spd/s-c6eda15c`
- Config file: `spd_out/spd/s-c6eda15c/final_config.yaml`
- `ci_alive_threshold`: `0.1`
- Toy model: TMS 5-2

---

## 1) Training Metrics (`train/*`)

### `train/loss/ImportanceMinimalityLoss`
- Meaning: sparsity pressure on CI values (encourages fewer active components).
- Better direction: lower (if reconstruction stays good).
- Code: `spd/metrics/importance_minimality_loss.py`

### `train/loss/StochasticReconLoss`
- Meaning: output reconstruction under stochastic masking across all decomposed layers.
- Better direction: lower.
- Code: `spd/metrics/stochastic_recon_loss.py`

### `train/loss/StochasticReconLayerwiseLoss`
- Meaning: output reconstruction under stochastic masking one layer at a time.
- Better direction: lower.
- Code: `spd/metrics/stochastic_recon_layerwise_loss.py`

### `train/loss/total`
- Meaning: weighted sum used for optimization.
- Formula source: `spd/losses.py` (`compute_total_loss`).

### `train/l0/linear1`, `train/l0/linear2`
- Meaning: average number of active components per sample in that layer (`CI > 0.1`).
- Better direction: task-dependent; for TMS, you usually want sparse-but-not-degenerate usage.
- Code:
  - `spd/utils/component_utils.py` (`calc_ci_l_zero`)
  - logged in `spd/run_spd.py`

### `train/grad_norms/...`
- Meaning: gradient norms for component parameters and CI-function parameters.
- Use: diagnose instability (exploding/vanishing), dead learning, imbalance.
- Code: `spd/utils/logging_utils.py` (`get_grad_norms_dict`)

### `train/schedules/lr`
- Meaning: current learning rate from schedule.
- Code: `spd/run_spd.py` and schedule logic in `spd/utils/general_utils.py` (`get_scheduled_value`)

---

## 2) Eval Metrics (`loss/*`, `l0/*`, `target_solution_error/*`)

These are computed during eval passes and appended into `metrics.jsonl`.

### `loss/FaithfulnessLoss`
- Meaning: MSE between target weights and sum of learned components (`W - UV`).
- Better direction: lower.
- Code: `spd/metrics/faithfulness_loss.py`

### `loss/UnmaskedReconLoss`
- Meaning: reconstruction loss when all components are fully on (mask = 1), no stochasticity.
- Better direction: lower.
- Code: `spd/metrics/unmasked_recon_loss.py`

### `loss/CIMaskedReconLoss`
- Meaning: reconstruction when masks are deterministic CI values (all layers together).
- Better direction: lower.
- Code: `spd/metrics/ci_masked_recon_loss.py`

### `loss/CIMaskedReconLayerwiseLoss`
- Meaning: same idea as above, but one layer masked at a time.
- Better direction: lower.
- Code: `spd/metrics/ci_masked_recon_layerwise_loss.py`

### `loss/StochasticHiddenActsReconLoss`
- Meaning: MSE between target hidden pre-weight activations and masked-model hidden pre-weight activations.
- Better direction: lower.
- Code: `spd/metrics/stochastic_hidden_acts_recon_loss.py`

### `loss/ImportanceMinimalityLoss`
- Meaning: eval version of CI sparsity penalty.
- Better direction: lower, but only meaningful alongside reconstruction metrics.
- Code: `spd/metrics/importance_minimality_loss.py`

### `loss/StochasticReconLoss`
- Meaning: eval version of stochastic output recon (all layers).
- Better direction: lower.
- Code: `spd/metrics/stochastic_recon_loss.py`

### `loss/StochasticReconLayerwiseLoss`
- Meaning: eval version of stochastic output recon (layerwise).
- Better direction: lower.
- Code: `spd/metrics/stochastic_recon_layerwise_loss.py`

### `l0/0.1_linear1`, `l0/0.1_linear2`
- Meaning: average active-component count per sample at threshold 0.1.
- Better direction: depends on decomposition objective; used to track minimality.
- Code: `spd/metrics/ci_l0.py`

### `target_solution_error/total`, `target_solution_error/total_0p2`, `target_solution_error/linear1`, `target_solution_error/linear2`
- Meaning: mismatch between learned CI structure and expected target CI pattern for toy model.
- Lower is better; `0` is perfect under the tolerance.
- `total_0p2` uses tolerance `0.2`; `total` uses `0.1`.
- Code:
  - `spd/metrics/identity_ci_error.py`
  - `spd/utils/target_ci_solutions.py`

---

## 3) Figure Artifacts (`spd_out/spd/s-c6eda15c/figures`)

### `figures_causal_importance_values_*.png`
- Histogram of lower-leaky CI values.
- Code: `spd/metrics/ci_histograms.py`, plotting in `spd/plotting.py` (`plot_ci_values_histograms`)

### `figures_causal_importance_values_pre_sigmoid_*.png`
- Histogram before sigmoid squashing (diagnose saturation/clipping).
- Code: same as above.

### `figures_causal_importances_*.png`, `figures_causal_importances_upper_leaky_*.png`
- Permuted CI matrix visualizations for one-hot feature probes.
- Code: `spd/metrics/permuted_ci_plots.py`, plotting in `spd/plotting.py` (`plot_causal_importance_vals`)

### `figures_component_activation_density_*.png`
- Per-component activation frequency.
- Code: `spd/metrics/component_activation_density.py`

### `figures_ci_mean_per_component_*.png`, `figures_ci_mean_per_component_log_*.png`
- Mean CI per component (linear/log scales).
- Code: `spd/metrics/ci_mean_per_component.py`

### `l0_bar_chart_*.json`
- Bar-chart payload for CI L0 by layer.
- Code: `spd/metrics/ci_l0.py` and write path in `spd/utils/logging_utils.py`

---

## 4) Alignment Report Metrics (`tms_alignment/*`)

From `koko_experiments/scripts/tms_alignment_report.py`:

### `MMCS`
- Mean max cosine similarity between target feature-direction columns and best-matching learned component columns.
- Closer to `1` is better.

### `ML2R`
- Mean ratio of matched learned column norm to target column norm.
- Closer to `1` is better (shrinkage/growth check).

### `coverage@0.99`, `coverage@0.95`
- Fraction of target columns whose best cosine similarity exceeds threshold.
- Higher is better.

### `faithfulness_mse`
- Layerwise weight reconstruction MSE.
- Lower is better.

### `scale_mae`
- Mean absolute deviation of `ML2R` from 1 at column level.
- Lower is better.

### `off_target_leakage`
- How much matched components also carry weight on non-target columns (crosstalk proxy).
- Lower is better.

### `effective_components@1pct_max_strength`
- Number of components above 1% of max strength (`||U_c||*||V_c||` heuristic).
- Use to estimate how many components are actually doing work.

---

## 5) Recommended Reading Order

1. `papers/Stochastic_Parameter_Decomposition/spd_paper.md` (Method + Appendix)
2. `spd/losses.py` (how train loss is combined)
3. `spd/eval.py` (how eval metric names are produced)
4. Individual metric modules under `spd/metrics/` listed above
5. `koko_experiments/scripts/tms_alignment_report.py` (extra toy-model geometry diagnostics)

