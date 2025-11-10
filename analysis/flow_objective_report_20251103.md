# Flow Objective Comparison (2025-11-03)

## Experiments
- `cfg/zinc-flow_baseline.yaml` (Rectified Flow) – log: `lgd_7937234.out` – run directory: `results/zinc_flow_20251103_170324/zinc-flow_baseline/0`
- `cfg/zinc-flow_gcfm.yaml` (Gaussian CFM) – log: `lgd_7937399.out` – run directory: `results/zinc_flow_20251103_175518/zinc-flow_gcfm/0`

Both jobs were launched via `scripts/zinc_trainer_diffusion_flow.sh`.

## Key Metrics at Best Validation Epoch
| Objective | Epoch | Val MAE | Val R² | Val Spearman | Test MAE | Test R² | Test Spearman |
|-----------|------:|--------:|-------:|-------------:|---------:|--------:|--------------:|
| Rectified | 109 | 0.381 | 0.926 | 0.966 | **0.376** | 0.944 | 0.972 |
| Gaussian CFM | 89 | 0.989 | 0.045 | 0.654 | 0.989 | 0.114 | 0.691 |

(Values rounded to three decimals; see `analysis/flow_objective_comparison_20251103.json` for exact numbers.)

## Observations
- **Curve shape (Rectified Flow):** validation MAE free-falls from 19 at epoch 0 to ≈0.39 by epoch 60, then plateaus in the 0.38–0.40 band through epoch 150. A shallow dip gives the best checkpoint at epoch 109 (0.381 val / 0.376 test), after which the curve drifts upward toward 0.41 by epoch 199—classic late-training flattening.
- **Curve shape (Gaussian CFM):** the run struggles to settle; MAE momentarily improves to 1.02 at epoch 49 and reaches the minimum 0.99 at epoch 89, but the curve never stabilises below 1.0. Past epoch 100 it rockets upward, exceeding 8.0 by epoch 180 and 9.5 by epoch 199, signalling objective collapse.
- Rectified flow thus remains stable and matches prior ZINC results, while Gaussian CFM, under current hyperparameters, diverges sharply.
- Splitting loss terms shows the Gaussian objective inflates `loss_task` while node/edge contributions stay comparable, suggesting the diffusion-style scaling in `cfg/zinc-flow_gcfm.yaml` needs re-tuning (e.g., different `sigma_{min,max}`, scheduler, or shorter warmup).

## Baseline DDPM Reference
- Previous diffusion baseline `results/zinc_diffusion_20250930_141105/zinc-diffusion_ddpm/0` (config `cfg/my_zinc-diffusion_ddpm_optimized.yaml`) still leads overall with test MAE 0.192 / R² 0.959 / Spearman 0.986 at epoch 249.
- Its validation curve drops from ≈1.6 at epoch 0 to ≈0.33 by epoch 100, plateaus between 0.33–0.36 through epoch 150, then continues improving under the long cosine schedule, delivering the final 0.217 val MAE at epoch 249.

## Suggested Follow-ups
1. Reduce `sigma_max` and increase `sigma_min` to tighten the Gaussian path; optionally try `sigma_fn: linear`.
2. Lower the learning rate warmup for Gaussian CFM or add gradient clipping specific to the task head.
3. Capture intermediate checkpoints (epochs 40–80) to inspect whether EMA weights improve stability before collapse.
