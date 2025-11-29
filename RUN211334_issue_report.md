Run: qm9_flow_20251118_210932 (wandb run-20251118_211334-0l693lvo, started 2025-11-18 21:13:35 JST)

Key observations
- Training advanced to epoch 29 (planned 100) then stopped without a completion message; last timestamps 23:09:31 JST. Only ckpt saved is 0.ckpt.
- Eval cadence (`eval_period=5`) ran at epochs 0/4/9/14/19/24; the eval scheduled for 29 never ran because the job ended right after logging the epoch-29 train stats.
- Val loss logged only at epoch 0 (0.63555331). Test loss logged at eval epochs but stays flat (~0.6429863), suggesting the eval loop/logging is reusing a constant value rather than recomputing with current weights.
- Training loss decreases (10.75 → 2.26 by epoch 29), so weights update, but generative metrics collapse: validity ~99% at epoch 4 → ~88% at 9 → ~48% at 14 → ~0.1–0.03% by 19–24. RDKit raises many “Explicit valence … greater than permitted” warnings during sampling.
- No stack trace; likely preempted/killed, not gracefully finished.

Files to inspect
- train_diffusion.py (top-level) and lgd/train/train_diffusion.py for training/eval loop.
- cfg/QM9_unconditional_generation_flow_gcfm.yaml and runs/qm9_flow_20251118_210932/QM9_unconditional_generation_flow_gcfm/config.yaml for resolved config.
- flow_train.log and wandb/run-20251118_211334-0l693lvo/files/output.log for metrics/warnings.

Potential root causes to check
- Eval loop might be reusing cached loss or not switching to eval dataloader, causing flat val/test losses.
- Sampling configuration/EMA usage may produce invalid molecules; inspect sampler settings and any valence constraints.
- Job termination before epoch 29 eval suggests wall-time/preemption; rerun with longer time or monitoring.
