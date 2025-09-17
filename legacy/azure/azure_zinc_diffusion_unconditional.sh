#!/bin/bash
# Legacy Azure wrapper: moved from scripts/
# Azure VM wrapper for ZINC UNCONDITIONAL diffusion training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/ops/env/azure.sh"

ENCODER_CHECKPOINT="${1:-auto}"
CONFIG="${2:-cfg/zinc-diffusion_ddpm_unconditional.yaml}"
REPEAT="${3:-5}"
MAX_EPOCH="${4:-50}"
EXP="zinc_uncond_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUNS/$EXP" "$DATA"
echo "[azure] Starting unconditional diffusion training: $EXP"

if command -v singularity &>/dev/null; then
  BIND_WANDB=""
  if [ -f "$CODE/.wandbrc" ]; then
    BIND_WANDB="-B \"$CODE/.wandbrc\":/workspace/.wandbrc"
  fi
  singularity exec --nv \
    -B "$CODE":/workspace \
    -B "$RUNS":/workspace/runs \
    -B "$DATA":/workspace/data \
    $BIND_WANDB \
    "$IMG" \
    bash -lc \
      "
        # Configure WandB from .wandbrc if present
        if [ -f /workspace/.wandbrc ]; then
          export WANDB_API_KEY=\$(awk -F' *= *' '/^api_key/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
          export WANDB_ENTITY=\$(awk -F' *= *' '/^entity/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
          export WANDB_MODE=online;
          echo \"WandB Entity: '\$WANDB_ENTITY'\";
          echo \"WandB API Key length: \${#WANDB_API_KEY}\";
        fi;

        /workspace/ops/runners/run_zinc_train_diffusion_uncond.sh \
        --checkpoint $ENCODER_CHECKPOINT \
        --config $CONFIG \
        --repeat $REPEAT \
        --max-epoch $MAX_EPOCH \
        --out-dir /workspace/runs/$EXP \
        --wandb-name zinc_uncond_$EXP"
else
  cd "$CODE"
  export PYTHONPATH=$CODE:$PYTHONPATH
  bash ops/runners/run_zinc_train_diffusion_uncond.sh \
    --checkpoint "$ENCODER_CHECKPOINT" \
    --config "$CONFIG" \
    --repeat "$REPEAT" \
    --max-epoch "$MAX_EPOCH" \
    --out-dir "$RUNS/$EXP" \
    --wandb-name "zinc_uncond_$EXP"
fi

echo "[azure] Done. Logs in $RUNS/$EXP"
