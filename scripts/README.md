Scripts Overview
================

This folder provides two building blocks:

1) Environment presets (scripts/env)
   - Machine-specific defaults for paths, networking, and WandB.
   - `scripts/env/wisteria.sh` (HPC), `scripts/env/azure.sh` (Azure VM).
   - Export `ROOT`, `CODE`, `DATA`, `RUNS`, and helpful NCCL/GLOO settings.

2) Inside-container runners (scripts/common)
   - Reusable shell entry points that run Python training with consistent
     arguments and logging inside `/workspace`.
   - ZINC:
     - `run_zinc_pretrain_encoder.sh`
     - `run_zinc_train_diffusion.sh`
     - `run_zinc_train_flow.sh`
     - `run_zinc_train_diffusion_uncond.sh`
   - QM9:
     - `run_qm9_pretrain_encoder.sh`
     - `run_qm9_train_diffusion.sh`

Usage Patterns
- From a job wrapper (recommended):
  - Wisteria PJM wrappers in `experiments/wisteria/*` call these runners inside
    the container with proper bind mounts.
  - Azure wrappers in `experiments/azure/*` do the same on Azure VMs.

- Directly inside the container (manual runs):
  - `bash scripts/common/run_zinc_train_diffusion.sh --checkpoint auto --config cfg/zinc-diffusion_ddpm.yaml --repeat 5 --max-epoch 50`
  - `bash scripts/common/run_zinc_train_flow.sh --checkpoint auto --config cfg/zinc-flow_rf.yaml --max-epoch 300`
  - `bash scripts/common/run_zinc_train_diffusion_uncond.sh --checkpoint auto`

Conventions
- Checkpoint auto-detection: pass `--checkpoint auto` to find the latest
  ZINC/QM9 encoder ckpt under `/workspace`.
- Output and logs: runners write to `/workspace/runs/<exp>` by default. YAML
  configs may also write to `results/` depending on `out_dir`.
- WandB: runners set `wandb.use True` but rely on the environment for API key.

Legacy Scripts
- Older, one-off training scripts are preserved in `scripts/legacy/` for
  reference. Prefer job wrappers in `experiments/*` and runners in
  `scripts/common/` going forward.

