#!/bin/bash
set -euo pipefail

# DEPRECATED: use experiments/azure/zinc_train_flow.sh instead
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "[DEPRECATED] scripts/azure_zinc_flow_training.sh -> experiments/azure/zinc_train_flow.sh" >&2
exec "$REPO_ROOT/experiments/azure/zinc_train_flow.sh" "$@"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export SEED=$RUN_ID

echo "Starting Latent Flow Matching Training"
echo "Config: $CONFIG"
echo "Encoder checkpoint: $ENCODER_CKPT"
echo "Max Epochs: $MAX_EPOCH"
echo "Experiment: $EXP"
echo "Run ID: $RUN_ID"
echo "Timestamp: $TIMESTAMP"
echo "Time started: $(date)"
echo ""
echo "=== Flow Matching Configuration ==="
echo "- Objective: Rectified Flow"
echo "- ODE Solver: Heun (RK2)"
echo "- NFE: 20 steps"
echo "- Pretrained encoder: $ENCODER_CKPT"
echo "- EMA enabled"
echo "- WandB logging: $WANDB_MODE"
echo ""

# -------- Check if singularity is available --------
if command -v singularity &> /dev/null; then
    echo "Using Singularity container"
    
    # Check if encoder checkpoint exists
    if [ ! -f "$CODE/$ENCODER_CKPT" ]; then
        echo "ERROR: Encoder checkpoint not found at $CODE/$ENCODER_CKPT"
        echo "Please ensure the encoder pretraining has been completed"
        exit 1
    fi
    
    # -------- singularity + LGD commands --------
    singularity exec --nv \
      -B "$CODE":/workspace \
      -B "$RUNS":/workspace/runs \
      -B "$DATA":/workspace/data \
      -B "$RESULTS":/workspace/results \
      -B "$CODE/.wandbrc":/workspace/.wandbrc \
      "$IMG" \
      bash -c "
        cd /workspace;
        export PYTHONPATH=/workspace:\$PYTHONPATH;
        export PYTHONUNBUFFERED=1;
        
        # Set WandB config from .wandbrc
        if [ -f /workspace/.wandbrc ]; then
          echo 'Found .wandbrc file, setting WandB config';
          export WANDB_API_KEY=\$(awk -F' *= *' '/^api_key/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
          export WANDB_ENTITY=\$(awk -F' *= *' '/^entity/ {print \$2}' /workspace/.wandbrc | head -1 | tr -d '\\r\\n');
          echo \"WandB Entity: '\$WANDB_ENTITY'\";
          echo \"WandB API Key set (length: \${#WANDB_API_KEY})\";
          # Debug: show first/last characters
          echo \"API Key starts with: '\${WANDB_API_KEY:0:5}' ends with: '\${WANDB_API_KEY: -5}'\";
        else
          echo 'WARNING: .wandbrc not found, WandB may not work';
        fi;
        
        # Create experiment directory
        mkdir -p /workspace/runs/$EXP;
        mkdir -p /workspace/results/zinc-flow-rf/$RUN_ID;
        
        echo 'Environment check...';
        python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA device count: {torch.cuda.device_count()}\")';
        python -c 'import torch_geometric; print(f\"PyG version: {torch_geometric.__version__}\")';
        
        # Verify encoder checkpoint
        echo '';
        echo 'Verifying encoder checkpoint...';
        python -c \"
import torch
ckpt = torch.load('/workspace/$ENCODER_CKPT', map_location='cpu')
if 'state_dict' in ckpt:
    print(f'Encoder checkpoint loaded successfully')
    print(f'Epoch: {ckpt.get(\"epoch\", \"unknown\")}')
    if 'val/mae' in ckpt:
        print(f'Val MAE: {ckpt[\"val/mae\"]:.4f}')
else:
    print('Warning: checkpoint structure may be different')
\"
        
        echo '';
        echo 'Starting Flow Matching training...';
        echo \"Command: python train_diffusion.py --cfg $CONFIG flow.first_stage_config /workspace/$ENCODER_CKPT optim.max_epoch $MAX_EPOCH wandb.use True\";
        
        python train_diffusion.py \
            --cfg $CONFIG \
            flow.first_stage_config /workspace/$ENCODER_CKPT \
            optim.max_epoch $MAX_EPOCH \
            wandb.use True \
            out_dir /workspace/results/zinc-flow-rf/$RUN_ID \
            2>&1 | tee /workspace/runs/$EXP/flow_training_full.log;
        
        # Check training results
        echo '';
        echo 'Training completed. Checking results...';
        find /workspace/results/zinc-flow-rf/$RUN_ID -name '*.ckpt' -exec ls -la {} \;
        
        # Find best checkpoint
        BEST_CKPT=\$(find /workspace/results/zinc-flow-rf/$RUN_ID -name '*best*.ckpt' -type f | head -1);
        if [ ! -z \"\$BEST_CKPT\" ]; then
            echo \"Best checkpoint found: \$BEST_CKPT\";
        fi
        
        # Summary
        echo '';
        echo '=== LGD Training Summary ===';
        echo 'Stage 1 (Encoder): $ENCODER_CKPT';
        echo \"Stage 2 (Flow): results/zinc-flow-rf/$RUN_ID/\";
        echo 'Use the flow model checkpoint for graph generation via ODE sampling';
        
        # Log completion time
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
