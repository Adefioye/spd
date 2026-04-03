# exp_03_tms_5_2_paper_diff_seeds

Five repeated SPD decompositions of the exact paper-style [tms_5-2_cpu_paper.yaml](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/parameter_recovery/experiments/exp_03_tms_5_2_paper_diff_seeds/tms_5-2_cpu_paper.yaml) with five different seeds and the same pretrained target model path.

Run from the repo root:

```bash
source .venv/bin/activate
python -m koko_experiments.parameter_recovery.run_tms_paper_replicates \
  koko_experiments/parameter_recovery/experiments/exp_03_tms_5_2_paper_diff_seeds
```
