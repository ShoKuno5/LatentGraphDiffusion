#!/bin/bash

# Azure Fast Training Script for ZINC Dataset
# Based on CLAUDE.md optimizations - uses zinc-encoder-fast.yaml config
# Runs both encoder pretraining and diffusion training in sequence

# -------- paths --------
CODE=/home/azureuser/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- experiment tag ---------
EXP="zinc_fast_$(date +%Y%m%d_%H%M%S)"
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

export WANDB_PROJECT=LatentGraphDiffusion-ZINC
export WANDB_NAME="zinc_fast_${EXP}"

# -------- job parameters --------
# Optimized parameters from CLAUDE.md
ENCODER_CONFIG="cfg/zinc-encoder-fast.yaml"
DIFFUSION_CONFIG="cfg/zinc-diffusion_ddpm.yaml"
ENCODER_EPOCHS=400   # Reduced from 2000
DIFFUSION_EPOCHS=50  # Can be further tuned
REPEAT=1             # Single run for fast iteration

echo "===== Starting ZINC Fast Training Pipeline ====="
echo "Experiment: $EXP"
echo "Encoder Config: $ENCODER_CONFIG"
echo "Encoder Epochs: $ENCODER_EPOCHS"
echo "Diffusion Config: $DIFFUSION_CONFIG"
echo "Diffusion Epochs: $DIFFUSION_EPOCHS"
echo "Time started: $(date)"
echo ""

# Function to run training either in singularity or local
run_training() {
    local phase=$1
    local config=$2
    local epochs=$3
    local extra_args=$4
    
    if command -v singularity &> /dev/null; then
        echo "Using Singularity container for $phase"
        singularity exec --nv \
          -B "$CODE":/workspace \
          -B "$RUNS":/workspace/runs \
          -B "$DATA":/workspace/data \
          "$IMG" \
          bash -c "
            cd /workspace;
            export PYTHONPATH=/workspace:\$PYTHONPATH;
            export PYTHONUNBUFFERED=1;
            
            if [ \"$phase\" = \"encoder\" ]; then
                python pretrain.py \
                    --cfg $config \
                    --repeat $REPEAT \
                    wandb.use True \
                    optim.max_epoch $epochs \
                    train.early_stop True \
                    train.early_stop_patience 20 \
                    out_dir /workspace/runs/$EXP/encoder \
                    2>&1 | tee /workspace/runs/$EXP/${phase}_train.log
            else
                python train_diffusion.py \
                    --cfg $config \
                    --repeat $REPEAT \
                    wandb.use True \
                    optim.max_epoch $epochs \
                    $extra_args \
                    out_dir /workspace/runs/$EXP/diffusion \
                    2>&1 | tee /workspace/runs/$EXP/${phase}_train.log
            fi
          "
    else
        echo "Running $phase in local environment"
        cd "$CODE"
        export PYTHONPATH=$CODE:$PYTHONPATH
        export PYTHONUNBUFFERED=1
        
        if [ "$phase" = "encoder" ]; then
            python pretrain.py \
                --cfg $config \
                --repeat $REPEAT \
                wandb.use True \
                optim.max_epoch $epochs \
                train.early_stop True \
                train.early_stop_patience 20 \
                out_dir "$RUNS/$EXP/encoder" \
                2>&1 | tee "$RUNS/$EXP/${phase}_train.log"
        else
            python train_diffusion.py \
                --cfg $config \
                --repeat $REPEAT \
                wandb.use True \
                optim.max_epoch $epochs \
                $extra_args \
                out_dir "$RUNS/$EXP/diffusion" \
                2>&1 | tee "$RUNS/$EXP/${phase}_train.log"
        fi
    fi
}

# ===== Phase 1: Encoder Pretraining =====
echo "===== Phase 1: Encoder Pretraining ====="
echo "Using optimized config with:"
echo "- Reduced epochs: 400 (from 2000)"
echo "- Early stopping: patience=20"
echo "- Smaller model: hid_dim=32, num_layers=6"
echo ""

run_training "encoder" "$ENCODER_CONFIG" "$ENCODER_EPOCHS"

# Check for encoder checkpoint
echo ""
echo "Looking for encoder checkpoint..."
ENCODER_CHECKPOINT=$(find "$EXP_DIR/encoder" -name "*.ckpt" | sort -V | tail -1)

if [ -z "$ENCODER_CHECKPOINT" ] || [ ! -f "$ENCODER_CHECKPOINT" ]; then
    echo "ERROR: Encoder pretraining failed - no checkpoint found"
    echo "Check logs at: $EXP_DIR/encoder_train.log"
    exit 1
fi

echo "Found encoder checkpoint: $ENCODER_CHECKPOINT"

# ===== Phase 2: Diffusion Training =====
echo ""
echo "===== Phase 2: Diffusion Training ====="
echo "Using encoder checkpoint from Phase 1"
echo ""

run_training "diffusion" "$DIFFUSION_CONFIG" "$DIFFUSION_EPOCHS" "diffusion.first_stage_config \"$ENCODER_CHECKPOINT\""

# ===== Summary =====
echo ""
echo "===== Training Complete ====="
echo "Total time: $(date)"
echo "Results saved in: $EXP_DIR"
echo ""
echo "Encoder checkpoints:"
find "$EXP_DIR/encoder" -name "*.ckpt" -type f | while read -r ckpt; do
    echo "  - $ckpt"
done
echo ""
echo "Diffusion checkpoints:"
find "$EXP_DIR/diffusion" -name "*.ckpt" -type f | while read -r ckpt; do
    echo "  - $ckpt"
done

# Create completion marker with summary
cat > "$EXP_DIR/training_summary.txt" << EOF
ZINC Fast Training Summary
========================
Experiment: $EXP
Started: $(date)
Encoder Config: $ENCODER_CONFIG
Encoder Epochs: $ENCODER_EPOCHS
Encoder Checkpoint: $ENCODER_CHECKPOINT
Diffusion Config: $DIFFUSION_CONFIG
Diffusion Epochs: $DIFFUSION_EPOCHS

Optimizations Applied:
- Reduced epochs from 2000 to 400 for encoder
- Early stopping with patience=20
- Smaller model architecture (hid_dim=32, layers=6)
- Using ZINC subset dataset (12k molecules)

Results Directory: $EXP_DIR
EOF

echo ""
echo "Training summary saved to: $EXP_DIR/training_summary.txt"
echo ""
echo "To view results in WandB:"
echo "  https://wandb.ai/$WANDB_ENTITY/LatentGraphDiffusion-ZINC"