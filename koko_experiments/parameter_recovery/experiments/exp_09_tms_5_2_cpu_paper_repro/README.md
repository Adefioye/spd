# exp_09: TMS 5-2 CPU Paper Reproduction

This experiment folder wraps the paper-style TMS 5-2 SPD config in
[`koko_experiments/configs/tms_5-2_cpu_paper.yaml`](/Users/abdulhakeemadefioye/Desktop/deep-learning/koko_spd/koko_experiments/configs/tms_5-2_cpu_paper.yaml)
with explicit `exp_09` W&B naming and a detached `tmux` launcher.

## What is here

- `run_exp_09_cpu_paper.py`: materialize a W&B-enabled SPD config and run the CPU decomposition.
- `launch_tmux.sh`: launch the CPU SPD run in a detached `tmux` session.

## Default W&B naming

- project: `spd_tms_5_2_exp09`
- run: `exp_09_tms_5_2_cpu_paper`

## Run in tmux

```bash
source .venv/bin/activate
bash koko_experiments/parameter_recovery/experiments/exp_09_tms_5_2_cpu_paper_repro/launch_tmux.sh
```

The launcher:

- runs on CPU
- writes logs under `logs/`
- materializes the final SPD config under `materialized_configs/`
- writes run metadata under `run_records/`
