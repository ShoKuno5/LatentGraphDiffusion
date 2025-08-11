#!/bin/bash

# Azure-compatible version of zinc_pretrain_encoder.sh
# Runs ZINC encoder pretraining on Azure VM

# -------- paths --------
CODE=/home/azureuser/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- experiment tag ---------
EXP="zinc_encoder_$(date +%Y%m%d_%H%M%S)"
EXP_DIR=$RUNS/$EXP
mkdir -p "$DATA" "$EXP_DIR"
echo "Directory created: $EXP_DIR $DATA"

# -------- env / NCCL / PyTorch --------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_MODE=offline
    echo "WANDB_API_KEY not set, using offline mode"
else
    export WANDB_MODE=online
fi

export WANDB_PROJECT=latentgraphdiffusion
export WANDB_NAME="zinc_encoder_${EXP}"

# -------- job parameters --------
CONFIG="${1:-cfg/zinc-encoder-fast.yaml}"  # Use fast config by default, allow override
REPEAT="${2:-5}"                           # Default 5 repeats
MAX_EPOCH="${3:-400}"                      # Default 400 epochs for fast config

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