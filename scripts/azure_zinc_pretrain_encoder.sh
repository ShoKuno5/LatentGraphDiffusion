#!/bin/bash
set -euo pipefail

# DEPRECATED: use experiments/azure/zinc_pretrain_encoder.sh instead
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "[DEPRECATED] scripts/azure_zinc_pretrain_encoder.sh -> experiments/azure/zinc_pretrain_encoder.sh" >&2
exec "$REPO_ROOT/experiments/azure/zinc_pretrain_encoder.sh" "$@"

echo "Starting ZINC Encoder Pretraining Job"
echo "Config: $CONFIG"
echo "Max Epochs: $MAX_EPOCH"
echo "Repeats: $REPEAT"
echo "Experiment: $EXP"
echo "Time started: $(date)"

# -------- Check if singularity is available --------
if command -v singularity &> /dev/null; then
    echo "Using Singularity container"
    # -------- singularity + LGD commands --------
    singularity exec --nv \
      -B "$CODE":/workspace \
      -B "$RUNS":/workspace/runs \
      -B "$DATA":/workspace/data \
      "$IMG" \
      bash -c "
        cd /workspace;
        export PYTHONPATH=/workspace:\$PYTHONPATH;
        export PYTHONUNBUFFERED=1;
        
        # Create experiment directory
        mkdir -p /workspace/runs/$EXP;
        
        echo 'Environment check...';
        python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA device count: {torch.cuda.device_count()}\")';
        python -c 'import torch_geometric; print(f\"PyG version: {torch_geometric.__version__}\")';
        
        echo 'Starting ZINC encoder pretraining...';
        echo 'Command: python pretrain.py --cfg $CONFIG --repeat $REPEAT wandb.use True optim.max_epoch $MAX_EPOCH';
        
        python pretrain.py \
            --cfg $CONFIG \
            --repeat $REPEAT \
            wandb.use True \
            optim.max_epoch $MAX_EPOCH \
            out_dir /workspace/runs/$EXP \
            2>&1 | tee /workspace/runs/$EXP/pretrain_full.log;
        
        # Check training results
        echo 'Training completed. Checking results...';
        find /workspace/runs/$EXP -name '*.ckpt' -exec ls -la {} \;
        
        # Log completion time
        echo 'Training completed at: $(date)';
        echo 'Results saved in: /workspace/runs/$EXP';
      "
else
    echo "Singularity not found, running in local environment"
    # -------- Run directly in local Python environment --------
    cd "$CODE"
    export PYTHONPATH=$CODE:$PYTHONPATH
    export PYTHONUNBUFFERED=1
    
    # Create experiment directory
    mkdir -p "$RUNS/$EXP"
    
    echo 'Environment check...'
    python -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA device count: {torch.cuda.device_count()}")'
    python -c 'import torch_geometric; print(f"PyG version: {torch_geometric.__version__}")'
    
    echo 'Starting ZINC encoder pretraining...'
    echo "Command: python pretrain.py --cfg $CONFIG --repeat $REPEAT wandb.use True optim.max_epoch $MAX_EPOCH"
    
    python pretrain.py \
        --cfg $CONFIG \
        --repeat $REPEAT \
        wandb.use True \
        optim.max_epoch $MAX_EPOCH \
        out_dir "$RUNS/$EXP" \
        2>&1 | tee "$RUNS/$EXP/pretrain_full.log"
    
    # Check training results
    echo 'Training completed. Checking results...'
    find "$RUNS/$EXP" -name '*.ckpt' -exec ls -la {} \;
    
    # Log completion time
    echo "Training completed at: $(date)"
    echo "Results saved in: $RUNS/$EXP"
fi

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create job completion marker
touch "$EXP_DIR/job_completed.txt"
echo "Job completed successfully at $(date)" > "$EXP_DIR/job_completed.txt"

# Print final checkpoint locations for easy reference
echo ""
echo "===== Final checkpoint locations ====="
find "$EXP_DIR" -name '*.ckpt' -type f | while read -r ckpt; do
    echo "Checkpoint: $ckpt"
done
