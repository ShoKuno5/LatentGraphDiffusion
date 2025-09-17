#!/bin/bash
# Legacy Azure wrapper: moved from scripts/
set -euo pipefail

# Azure VM wrapper for ZINC encoder pretraining (canonical)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/ops/env/azure.sh"

CONFIG="${1:-cfg/zinc-encoder.yaml}"
REPEAT="${2:-5}"
MAX_EPOCH="${3:-50}"
EXP="zinc_encoder_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUNS/$EXP" "$DATA"
echo "[azure] Starting encoder pretraining: $EXP"

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
      "/workspace/ops/runners/run_zinc_pretrain_encoder.sh \
        --config $CONFIG \
        --repeat $REPEAT \
        --max-epoch $MAX_EPOCH \
        --out-dir /workspace/runs/$EXP \
        --wandb-name zinc_encoder_$EXP"
else
  cd "$CODE"
  export PYTHONPATH=$CODE:$PYTHONPATH
  bash ops/runners/run_zinc_pretrain_encoder.sh \
    --config "$CONFIG" \
    --repeat "$REPEAT" \
    --max-epoch "$MAX_EPOCH" \
    --out-dir "$RUNS/$EXP" \
    --wandb-name "zinc_encoder_$EXP"
fi

echo "[azure] Done. Logs in $RUNS/$EXP"
