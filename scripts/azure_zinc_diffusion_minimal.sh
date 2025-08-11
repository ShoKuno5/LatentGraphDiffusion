#!/bin/bash

# Minimal ZINC Diffusion Training - Just enough to verify it works
# Uses very few epochs to conserve compute resources

# -------- paths --------
CODE=/home/azureuser/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- Use the HPC checkpoint --------
ENCODER_CHECKPOINT="/home/azureuser/LatentGraphDiffusion/runs/zinc_encoder_fast_hpc/zinc-encoder-fast/0/ckpt/399.ckpt"
CONFIG="cfg/zinc-diffusion_ddpm.yaml"

# -------- Minimal training parameters --------
REPEAT=1          # Single run only
MAX_EPOCH=10      # Just 10 epochs to test
CKPT_PERIOD=5     # Save checkpoint every 5 epochs

# -------- experiment tag ---------
EXP="zinc_diffusion_minimal_$(date +%Y%m%d_%H%M%S)"
EXP_DIR=$RUNS/$EXP
mkdir -p "$DATA" "$EXP_DIR"
echo "Directory created: $EXP_DIR"

# -------- env / NCCL / PyTorch --------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8

# WandB configuration - extract API key from .wandbrc (with proper whitespace trimming)
WANDB_API_KEY=$(grep "api_key" /home/azureuser/.wandbrc | head -n1 | cut -d'=' -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
export WANDB_API_KEY
export WANDB_ENTITY="shokuno-the-university-of-tokyo"
export WANDB_PROJECT="LatentGraphDiffusion-ZINC"
export WANDB_NAME="zinc_diffusion_minimal_${EXP}"

echo "===== Minimal ZINC Diffusion Training ====="
echo "Checkpoint: $ENCODER_CHECKPOINT"
echo "Config: $CONFIG"
echo "Max Epochs: $MAX_EPOCH (minimal test run)"
echo "Experiment: $EXP"
echo "Time started: $(date)"
echo ""

# Verify checkpoint exists
if [ ! -f "$ENCODER_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at: $ENCODER_CHECKPOINT"
    exit 1
fi
echo "✓ Checkpoint verified"

# -------- Check if singularity is available --------
if command -v singularity &> /dev/null; then
    echo "Using Singularity container"
    # -------- singularity + LGD commands --------
    singularity exec --nv \
      --env WANDB_API_KEY="$WANDB_API_KEY" \
      --env WANDB_ENTITY="$WANDB_ENTITY" \
      --env WANDB_PROJECT="$WANDB_PROJECT" \
      --env WANDB_NAME="$WANDB_NAME" \
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
        
        echo 'Starting minimal diffusion training...';
        python train_diffusion.py \
            --cfg $CONFIG \
            --repeat $REPEAT \
            wandb.use True \
            wandb.entity \"$WANDB_ENTITY\" \
            wandb.project \"$WANDB_PROJECT\" \
            wandb.name \"$WANDB_NAME\" \
            optim.max_epoch $MAX_EPOCH \
            train.ckpt_period $CKPT_PERIOD \
            diffusion.first_stage_config \"$ENCODER_CHECKPOINT\" \
            out_dir /workspace/runs/$EXP \
            2>&1 | tee /workspace/runs/$EXP/diffusion_minimal.log;
        
        echo '';
        echo 'Training completed. Checking results...';
        find /workspace/runs/$EXP -name '*.ckpt' -exec ls -la {} \;
      "
else
    echo "Running in local environment"
    # -------- Run directly in local Python environment --------
    cd "$CODE"
    export PYTHONPATH=$CODE:$PYTHONPATH
    export PYTHONUNBUFFERED=1
    
    # Create experiment directory
    mkdir -p "$RUNS/$EXP"
    
    echo 'Starting minimal diffusion training...'
    python train_diffusion.py \
        --cfg $CONFIG \
        --repeat $REPEAT \
        wandb.use True \
        wandb.entity "$WANDB_ENTITY" \
        wandb.project "$WANDB_PROJECT" \
        wandb.name "$WANDB_NAME" \
        optim.max_epoch $MAX_EPOCH \
        train.ckpt_period $CKPT_PERIOD \
        diffusion.first_stage_config "$ENCODER_CHECKPOINT" \
        out_dir "$RUNS/$EXP" \
        2>&1 | tee "$RUNS/$EXP/diffusion_minimal.log"
    
    echo ''
    echo 'Training completed. Checking results...'
    find "$RUNS/$EXP" -name '*.ckpt' -exec ls -la {} \;
fi

echo ""
echo "===== Minimal Test Complete ====="
echo "Time completed: $(date)"
echo "Results saved in: $EXP_DIR"
echo "Log file: $EXP_DIR/diffusion_minimal.log"
echo ""
echo "This was a minimal test run with only $MAX_EPOCH epochs."
echo "For full training, use azure_zinc_train_diffusion.sh with more epochs."