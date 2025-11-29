# LatentGraphDiffusion – pjsub コマンド早見表

## QM9 Unconditional パイプライン

1. **エンコーダ学習**（W&B project: `LGD-QM9-Uncond-Encoder`）  
   ```bash
   pjsub -x CONFIG=cfg/QM9_unconditional_generation_encoder.yaml \
         -x REPEAT=6 \
         -x MAX_EPOCH=500 \
         scripts/qm9_trainer_uncond_encoder.sh
   ```
2. **Diffusion（simple / complex）**（W&B project: `LGD-QM9-Uncond-Diffusion`）  
   ```bash
   # Simple
   pjsub -x CONFIG=cfg/QM9_unconditional_generation_diffusion_simple.yaml \
         -x CHECKPOINT=results/QM9_unconditional_generation_encoder/5/ckpt/209.ckpt \
         scripts/qm9_trainer_uncond_diffusion.sh

   # Complex
   pjsub -x CONFIG=cfg/QM9_unconditional_generation_diffusion_complex.yaml \
         -x CHECKPOINT=results/QM9_unconditional_generation_encoder/5/ckpt/209.ckpt \
         scripts/qm9_trainer_uncond_diffusion.sh
   ```
   `CHECKPOINT=auto` のままにすると `/workspace` 以下から最新の QM9 無条件エンコーダ ckpt を自動検出します。
3. **Flow Matching**（W&B project: `LGD-QM9-Uncond-Flow`）  
   ```bash
   pjsub -x CONFIG=cfg/QM9_unconditional_generation_flow.yaml \
         -x CHECKPOINT=runs/qm9_uncond_encoder_20251117_090511/QM9_unconditional_generation_encoder/0/ckpt/499.ckpt
         -x MAX_EPOCH=100 \
         scripts/qm9_trainer_flow.sh
   ```
   Flow も `CHECKPOINT=auto` で自動探索可能。`MAX_EPOCH` や `WANDB_NAME` などは `-x KEY=VALUE` で上書きしてください。

## ZINC パイプライン

1. **エンコーダ学習**（W&B project: `LGD-ZINC-Encoder`）  
   ```bash
   pjsub -x CONFIG=cfg/zinc-encoder.yaml \
         -x REPEAT=5 \
         -x MAX_EPOCH=50 \
         scripts/zinc_trainer_encoder.sh
   ```
2. **Diffusion（条件付き DDPM）**（W&B project: `LGD-ZINC-Diffusion`）  
   ```bash
   pjsub -x CONFIG=cfg/zinc-diffusion_ddpm.yaml \
         -x CHECKPOINT=/workspace/runs/zinc_encoder_20250930_004610/.../ckpt/399.ckpt \
         -x REPEAT=5 \
         -x MAX_EPOCH=50 \
         scripts/zinc_trainer_diffusion.sh
   ```
3. **無条件 Diffusion**（W&B project: `LGD-ZINC-Unconditional`）  
   ```bash
   pjsub -x CONFIG=cfg/zinc-diffusion_ddpm_unconditional.yaml \
         -x CHECKPOINT=/workspace/runs/zinc_encoder_20250930_004610/.../ckpt/399.ckpt \
         scripts/zinc_trainer_uncond.sh
   ```
4. **Flow Matching**（W&B project: `LGD-ZINC-Flow`）  
   ```bash
   pjsub -x CONFIG=cfg/zinc-flow_baseline.yaml \
         -x CHECKPOINT=/workspace/runs/zinc_encoder_20250930_004610/.../ckpt/399.ckpt \
         -x MAX_EPOCH=300 \
         scripts/zinc_trainer_flow.sh
   ```

※ 以前の統合スクリプト `scripts/zinc_trainer_diffusion_flow.sh` も引き続き利用できます（`DATASET` / `MODE` で制御）。

- **エンコーダ学習**（W&B project: `LGD-ZINC-Encoder`）  
   ```bash
   pjsub -x DATASET=zinc -x MODE=encoder \
         -x CONFIG=cfg/zinc-encoder.yaml \
         scripts/zinc_trainer_diffusion_flow.sh
   ```
- **Diffusion（条件付き / 無条件）**  
   ```bash
   # 条件付き DDPM
   pjsub -x DATASET=zinc -x MODE=diffusion \
         -x CONFIG=cfg/zinc-diffusion_ddpm.yaml \
         -x CHECKPOINT=/workspace/runs/zinc_encoder_20250930_004610/.../ckpt/399.ckpt \
         scripts/zinc_trainer_diffusion_flow.sh

   # 無条件 DDPM
   pjsub -x DATASET=zinc -x MODE=uncond \
         -x CONFIG=cfg/zinc-diffusion_ddpm_unconditional.yaml \
         -x CHECKPOINT=/workspace/runs/zinc_encoder_20250930_004610/.../ckpt/399.ckpt \
         scripts/zinc_trainer_diffusion_flow.sh
   ```
- **Flow Matching**  
   ```bash
   pjsub -x DATASET=zinc -x MODE=flow \
         -x CONFIG=cfg/zinc-flow_baseline.yaml \
         -x CHECKPOINT=/workspace/runs/zinc_encoder_20250930_004610/.../ckpt/399.ckpt \
         -x MAX_EPOCH=2000 \
         scripts/zinc_trainer_diffusion_flow.sh
   ```

### メモ
- `pjsub` のリソース指定（`-L rscgrp=...` など）は環境/キューに合わせてコマンドに追加してください。
- すべてのスクリプトは `.wandbrc` を自動バインドしつつ `wandb.use True` 固定で走ります。`-x WANDB_PROJECT=...` や `-x WANDB_NAME=...` を指定すればデフォルト値を上書きできます。
- `CHECKPOINT=auto` を指定するとスクリプトが `/workspace` 配下の最新 ckpt を探索します。明示的にパスを渡す場合は `-x CHECKPOINT=/path/to/ckpt/XYZ.ckpt` を指定してください。
