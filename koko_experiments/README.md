# Koko SPD Experiments: Toy Superposition (CPU First)

This folder gives a practical, step-by-step workflow for learning SPD on the toy model of superposition (`TMS 5-2`) on CPU.

## 1) Environment Setup (CPU)

From repository root:

```bash
cd /Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd

# keep SPD outputs inside this repo
export SPD_OUT_DIR="$PWD/spd_out"

# create/update local project files
make copy-templates

# install python deps with CPU-only pytorch wheels
uv venv --python 3.13
uv sync
# uv sync --frozen --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# frontend deps for the app
make install-app
```

## 2) Run TMS Superposition Decomposition (CPU)

Start with a shorter run:

```bash
CUDA_VISIBLE_DEVICES="" uv run python spd/experiments/tms/tms_decomposition.py \
  koko_experiments/configs/tms_5-2_cpu_quick.yaml
```

Paper-length run (same setup, longer):

```bash
CUDA_VISIBLE_DEVICES="" uv run python spd/experiments/tms/tms_decomposition.py \
  koko_experiments/configs/tms_5-2_cpu_paper.yaml
```

Notes:
- These configs keep `wandb_project: null` (no logging required).
- The target toy model is loaded from `pretrained_model_path` (W&B artifact path), so internet access is required unless already cached in `$SPD_OUT_DIR/runs/`.

## 3) Locate the Latest SPD Run Output

```bash
RUN_DIR=$(ls -td "$SPD_OUT_DIR"/spd/s-* | head -n 1)
echo "$RUN_DIR"
ls "$RUN_DIR"
```

Useful files:
- `final_config.yaml`
- `model_<step>.pth`
- `metrics.jsonl`
- `figures/` (CI plots, histograms, activation density, etc.)

## 4) Compute TMS Alignment Metrics (Paper + Extra)

Run:

```bash
uv run python koko_experiments/scripts/tms_alignment_report.py --run-dir "$RUN_DIR"
```

This prints:
- Paper metrics:
  - `MMCS` (Mean Max Cosine Similarity)
  - `ML2R` (Mean L2 Ratio)
- Additional useful metrics:
  - `faithfulness_mse` (weight reconstruction error)
  - `coverage@0.99`, `coverage@0.95` (column direction recovery rate)
  - `scale_mae` (magnitude mismatch)
  - `off_target_leakage` (component purity / crosstalk)
  - `effective_components@1pct_max_strength` (how many components are actually doing work)

## 5) Additional Metrics Worth Tracking (Beyond Paper)

The quick and paper configs in `configs/` already include these eval metrics:
- `FaithfulnessLoss`: checks if components sum back to target weights.
- `UnmaskedReconLoss`: output-level reconstruction when using summed components.
- `CIMaskedReconLoss`: reconstruction under CI-guided deterministic masking.
- `CIMaskedReconLayerwiseLoss`: same as above, layerwise.
- `StochasticHiddenActsReconLoss`: hidden activation reconstruction quality (internal faithfulness).
- `CI_L0`: sparsity/minimality of active components per input.
- `ComponentActivationDensity`: usage frequency per component.
- `CIMeanPerComponent`: dead/alive and over-active component diagnostics.
- `CIHistograms`: CI saturation/dead-zone checks.

For logs:

```bash
tail -n 5 "$RUN_DIR/metrics.jsonl"
```

## 6) Run the App Locally (CPU)

```bash
CUDA_VISIBLE_DEVICES="" make app
```

Then open the printed frontend URL (usually `http://localhost:5173`).

## 7) About Toy-Model Analysis in the App

Current app flow is tokenizer/prompt-centric (language-model runs). In the current codebase, TMS toy runs are not first-class in the app load path.

Practical workflow for TMS:
- Use the CPU TMS runs above.
- Use `metrics.jsonl`, `figures/`, and `tms_alignment_report.py` for decomposition analysis.

App familiarity workflow:
- Launch app with the command above.
- Load one of the canonical LM runs shown in the app UI (dropdown) to explore the interface and attribution tooling.
