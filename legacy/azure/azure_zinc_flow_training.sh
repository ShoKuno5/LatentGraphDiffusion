#!/bin/bash
# Legacy Azure wrapper: moved from scripts/
set -euo pipefail

# Azure VM wrapper for ZINC flow matching (canonical)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/ops/env/azure.sh"

ENCODER_CHECKPOINT="${1:-auto}"
CONFIG="${2:-cfg/zinc-flow_rf.yaml}"
MAX_EPOCH="${3:-300}"
EXP="zinc_flow_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUNS/$EXP" "$DATA"
echo "[azure] Starting flow matching: $EXP"

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
      "/workspace/ops/runners/run_zinc_train_flow.sh \
        --checkpoint $ENCODER_CHECKPOINT \
        --config $CONFIG \
        --max-epoch $MAX_EPOCH \
        --out-dir /workspace/runs/$EXP \
        --wandb-name zinc_flow_$EXP"
else
  cd "$CODE"
  export PYTHONPATH=$CODE:$PYTHONPATH
  bash ops/runners/run_zinc_train_flow.sh \
    --checkpoint "$ENCODER_CHECKPOINT" \
    --config "$CONFIG" \
    --max-epoch "$MAX_EPOCH" \
    --out-dir "$RUNS/$EXP" \
    --wandb-name "zinc_flow_$EXP"
fi

echo "[azure] Done. Logs in $RUNS/$EXP"
        echo '';
        echo \"Training completed at: \$(date)\";
        echo \"Results saved in: /workspace/results/zinc-flow-rf/$RUN_ID\";
      "
else
    echo "Singularity not found, running in local environment"
    # -------- Run directly in local Python environment --------
    cd "$CODE"
    export PYTHONPATH=$CODE:$PYTHONPATH
    export PYTHONUNBUFFERED=1
    
    # Create experiment directory
    mkdir -p "$RUNS/$EXP"
    mkdir -p "$RESULTS/zinc-flow-rf/$RUN_ID"
    
    # Check if encoder checkpoint exists
    if [ ! -f "$ENCODER_CKPT" ]; then
        echo "ERROR: Encoder checkpoint not found at $ENCODER_CKPT"
        echo "Please ensure the encoder pretraining has been completed"
        exit 1
    fi
    
    echo 'Environment check...'
    python -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA device count: {torch.cuda.device_count()}")'
    python -c 'import torch_geometric; print(f"PyG version: {torch_geometric.__version__}")'
    
    # Verify encoder checkpoint
    echo ''
    echo 'Verifying encoder checkpoint...'
    python -c "
import torch
ckpt = torch.load('$ENCODER_CKPT', map_location='cpu')
if 'state_dict' in ckpt:
    print(f'Encoder checkpoint loaded successfully')
    print(f'Epoch: {ckpt.get(\"epoch\", \"unknown\")}')
    if 'val/mae' in ckpt:
        print(f'Val MAE: {ckpt[\"val/mae\"]:.4f}')
else:
    print('Warning: checkpoint structure may be different')
"
    
    echo ''
    echo 'Starting Flow Matching training...'
    echo "Command: python train_diffusion.py --cfg $CONFIG flow.first_stage_config $ENCODER_CKPT optim.max_epoch $MAX_EPOCH wandb.use True"
    
    python train_diffusion.py \
        --cfg $CONFIG \
        flow.first_stage_config "$ENCODER_CKPT" \
        optim.max_epoch $MAX_EPOCH \
        wandb.use True \
        out_dir "$RESULTS/zinc-flow-rf/$RUN_ID" \
        2>&1 | tee "$RUNS/$EXP/flow_training_full.log"
    
    # Check training results
    echo ''
    echo 'Training completed. Checking results...'
    find "$RESULTS/zinc-flow-rf/$RUN_ID" -name '*.ckpt' -exec ls -la {} \;
    
    # Find best checkpoint
    BEST_CKPT=$(find "$RESULTS/zinc-flow-rf/$RUN_ID" -name '*best*.ckpt' -type f | head -1)
    if [ ! -z "$BEST_CKPT" ]; then
        echo "Best checkpoint found: $BEST_CKPT"
    fi
    
    # Summary
    echo ''
    echo '=== LGD Training Summary ==='
    echo "Stage 1 (Encoder): $ENCODER_CKPT"
    echo "Stage 2 (Flow): $RESULTS/zinc-flow-rf/$RUN_ID/"
    echo 'Use the flow model checkpoint for graph generation via ODE sampling'
    
    # Log completion time
    echo "Training completed at: $(date)"
    echo "Results saved in: $RESULTS/zinc-flow-rf/$RUN_ID"
fi

echo ""
echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create job completion marker
touch "$EXP_DIR/job_completed.txt"
echo "Job completed successfully at $(date)" > "$EXP_DIR/job_completed.txt"

# Print final checkpoint locations for easy reference
echo ""
echo "===== Final checkpoint locations ====="
find "$RESULTS/zinc-flow-rf/$RUN_ID" -name '*.ckpt' -type f 2>/dev/null | while read -r ckpt; do
    echo "Checkpoint: $ckpt"
done
