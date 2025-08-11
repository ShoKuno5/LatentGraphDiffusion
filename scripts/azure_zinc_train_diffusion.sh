#!/bin/bash

# Azure-compatible version of zinc_train_diffusion.sh
# Runs ZINC diffusion model training on Azure VM

# -------- paths --------
CODE=/home/azureuser/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- job parameters --------
# You can specify a checkpoint path, otherwise it will auto-detect
ENCODER_CHECKPOINT="${1:-auto}"  # Pass checkpoint path as first argument, or use 'auto'
CONFIG="${2:-cfg/zinc-diffusion_ddpm.yaml}"
REPEAT="${3:-5}"
MAX_EPOCH="${4:-50}"

# -------- experiment tag ---------
EXP="zinc_diffusion_$(date +%Y%m%d_%H%M%S)"
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
export WANDB_NAME="zinc_diffusion_${EXP}"

echo "Starting ZINC Diffusion Training Job"
echo "Config: $CONFIG"
echo "Max Epochs: $MAX_EPOCH"
echo "Repeats: $REPEAT"
echo "Encoder Checkpoint: $ENCODER_CHECKPOINT"
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
        
        # Auto-detect checkpoint if needed
        if [ \"$ENCODER_CHECKPOINT\" = \"auto\" ]; then
            echo 'Auto-detecting latest encoder checkpoint...';
            CHECKPOINT_PATH=\$(find /workspace/runs -path \"*/zinc-encoder/*/ckpt/*.ckpt\" -o -path \"*/zinc_encoder_*/ckpt/*.ckpt\" -o -path \"*/zinc_encoder_*/*.ckpt\" | sort -V | tail -1);
            if [ -z \"\$CHECKPOINT_PATH\" ]; then
                echo 'ERROR: No encoder checkpoint found. Please run encoder pretraining first or specify checkpoint path.';
                echo 'Available checkpoints:';
                find /workspace/runs -name \"*.ckpt\" | head -10;
                exit 1;
            fi;
        else
            CHECKPOINT_PATH=\"$ENCODER_CHECKPOINT\";
        fi;
        
        echo \"Using encoder checkpoint: \$CHECKPOINT_PATH\";
        
        # Verify checkpoint exists
        if [ ! -f \"\$CHECKPOINT_PATH\" ]; then
            echo 'ERROR: Checkpoint file does not exist: '\$CHECKPOINT_PATH'';
            exit 1;
        fi;
        
        echo 'Starting ZINC diffusion training...';
        echo \"Command: python train_diffusion.py --cfg $CONFIG --repeat $REPEAT wandb.use True optim.max_epoch $MAX_EPOCH diffusion.first_stage_config \\\"\$CHECKPOINT_PATH\\\"\";
        
        python train_diffusion.py \
            --cfg $CONFIG \
            --repeat $REPEAT \
            wandb.use True \
            optim.max_epoch $MAX_EPOCH \
            diffusion.first_stage_config \"\$CHECKPOINT_PATH\" \
            out_dir /workspace/runs/$EXP \
            2>&1 | tee /workspace/runs/$EXP/diffusion_train.log;
        
        # Check training results
        echo 'Training completed. Checking results...';
        find /workspace/runs/$EXP -name '*.ckpt' -exec ls -la {} \;
        
        # Log completion time
        echo \"Training completed at: \$(date)\";
        echo \"Results saved in: /workspace/runs/$EXP\";
        echo \"Used encoder checkpoint: \$CHECKPOINT_PATH\";
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
    
    # Auto-detect checkpoint if needed
    if [ "$ENCODER_CHECKPOINT" = "auto" ]; then
        echo 'Auto-detecting latest encoder checkpoint...'
        CHECKPOINT_PATH=$(find "$RUNS" -path "*/zinc-encoder/*/ckpt/*.ckpt" -o -path "*/zinc_encoder_*/ckpt/*.ckpt" -o -path "*/zinc_encoder_*/*.ckpt" | sort -V | tail -1)
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo 'ERROR: No encoder checkpoint found. Please run encoder pretraining first or specify checkpoint path.'
            echo 'Available checkpoints:'
            find "$RUNS" -name "*.ckpt" | head -10
            exit 1
        fi
    else
        CHECKPOINT_PATH="$ENCODER_CHECKPOINT"
    fi
    
    echo "Using encoder checkpoint: $CHECKPOINT_PATH"
    
    # Verify checkpoint exists
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Checkpoint file does not exist: $CHECKPOINT_PATH"
        exit 1
    fi
    
    echo 'Starting ZINC diffusion training...'
    echo "Command: python train_diffusion.py --cfg $CONFIG --repeat $REPEAT wandb.use True optim.max_epoch $MAX_EPOCH diffusion.first_stage_config \"$CHECKPOINT_PATH\""
    
    python train_diffusion.py \
        --cfg $CONFIG \
        --repeat $REPEAT \
        wandb.use True \
        optim.max_epoch $MAX_EPOCH \
        diffusion.first_stage_config "$CHECKPOINT_PATH" \
        out_dir "$RUNS/$EXP" \
        2>&1 | tee "$RUNS/$EXP/diffusion_train.log"
    
    # Check training results
    echo 'Training completed. Checking results...'
    find "$RUNS/$EXP" -name '*.ckpt' -exec ls -la {} \;
    
    # Log completion time
    echo "Training completed at: $(date)"
    echo "Results saved in: $RUNS/$EXP"
    echo "Used encoder checkpoint: $CHECKPOINT_PATH"
fi

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create job completion marker
touch "$EXP_DIR/job_completed.txt"
echo "Job completed successfully at $(date)" > "$EXP_DIR/job_completed.txt"
echo "Used encoder checkpoint: $ENCODER_CHECKPOINT" >> "$EXP_DIR/job_completed.txt"

# Print final checkpoint locations for easy reference
echo ""
echo "===== Final diffusion model checkpoints ====="
find "$EXP_DIR" -name '*.ckpt' -type f | while read -r ckpt; do
    echo "Checkpoint: $ckpt"
done