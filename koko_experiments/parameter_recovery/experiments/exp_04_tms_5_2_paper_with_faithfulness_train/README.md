# exp_04_tms_5_2_paper_with_faithfulness_train

Single paper-style TMS 5-2 SPD run using the same Goodfire pretrained target model as
[`koko_experiments/configs/tms_5-2_cpu_paper.yaml`](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/configs/tms_5-2_cpu_paper.yaml),
but with `FaithfulnessLoss` added to `loss_metric_configs` during main SPD training.

Run from the repo root:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_tms_single_with_analysis \
  koko_experiments/parameter_recovery/experiments/exp_04_tms_5_2_paper_with_faithfulness_train/tms_5-2_cpu_paper_with_faithfulness_train.yaml
```

This command:

- runs the SPD training job
- writes `parameter_recovery_analysis.json` into the SPD run directory
- writes only the rich analysis JSON into `SPD_OUT_DIR/parameter_recovery/results/...`
- copies the exact config used into the result bundle
