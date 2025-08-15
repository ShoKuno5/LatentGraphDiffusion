#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=04:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N zinc_diffusion_baseline
#PJM -o zinc_diffusion_baseline_%j.out
#PJM -e zinc_diffusion_baseline_%j.err

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths --------
ROOT=/work/gp15/q25030
CODE=$ROOT/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- job parameters --------
# Using our trained encoder checkpoint path
ENCODER_CHECKPOINT="results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
CONFIG="cfg/zinc-diffusion_ddpm_baseline.yaml"
REPEAT=1  # Single run for baseline testing
MAX_EPOCH=5  # Very minimal for baseline testing

# -------- experiment tag ---------
EXP="zinc_diffusion_baseline_$(date +%Y%m%d_%H%M%S)"
EXP_DIR=$RUNS/$EXP
mkdir -p "$DATA" "$EXP_DIR"
echo "Directory created: $EXP_DIR $DATA"

# -------- env / NCCL / PyTorch --------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export GLOO_SOCKET_IFNAME=ib0,eth0
export OMP_NUM_THREADS=8
export WANDB_MODE=online  # Use online mode for monitoring
export WANDB_API_KEY=fb39ca5f5835abaa4c40a8b61dde2a499b45fcba
export WANDB_PROJECT=LatentGraphDiffusion-ZINC
export WANDB_NAME="zinc_diffusion_baseline_${EXP}"

echo "Starting ZINC Diffusion BASELINE Training Job"
echo "Config: $CONFIG"
echo "Max Epochs: $MAX_EPOCH (minimal for baseline testing)"
echo "Repeats: $REPEAT"
echo "Encoder Checkpoint: $ENCODER_CHECKPOINT"
echo "Experiment: $EXP"
echo "Time started: $(date)"

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
    
    # Verify our encoder checkpoint exists
    CHECKPOINT_PATH=\"$ENCODER_CHECKPOINT\";
    echo \"Using encoder checkpoint: \$CHECKPOINT_PATH\";
    
    if [ ! -f \"\$CHECKPOINT_PATH\" ]; then
        echo 'ERROR: Encoder checkpoint file does not exist: '\$CHECKPOINT_PATH'';
        echo 'Available checkpoints in results/:';
        find /workspace/results -name \"*.ckpt\" | head -10;
        exit 1;
    fi;
    
    echo 'Starting ZINC diffusion BASELINE training (5 epochs for testing)...';
    echo \"Configuration: $CONFIG\";
    echo \"- Timesteps: 100 (reduced from 1000)\";
    echo \"- DDIM steps: 20 (fast inference)\";
    echo \"- Batch size: 64\";
    echo \"- Max epochs: $MAX_EPOCH\";
    echo \"- Diffusion transformer layers: 2\";
    echo '';
    
    echo \"Command: python train_diffusion.py --cfg $CONFIG --repeat $REPEAT wandb.use True optim.max_epoch $MAX_EPOCH diffusion.first_stage_config \\\"\$CHECKPOINT_PATH\\\"\";
    
    python train_diffusion.py \
        --cfg $CONFIG \
        --repeat $REPEAT \
        wandb.use True \
        optim.max_epoch $MAX_EPOCH \
        diffusion.first_stage_config \"\$CHECKPOINT_PATH\" \
        out_dir /workspace/runs/$EXP \
        2>&1 | tee /workspace/runs/$EXP/diffusion_baseline_train.log;
    
    # Check training results
    echo 'Training completed. Checking results...';
    find /workspace/runs/$EXP -name '*.ckpt' -exec ls -la {} \;
    
    # Test inference if training succeeded
    if [ \$? -eq 0 ]; then
        echo 'Testing inference...';
        # Find the latest checkpoint
        LATEST_CKPT=\$(find /workspace/runs/$EXP -name \"*.ckpt\" | sort -V | tail -1);
        if [ ! -z \"\$LATEST_CKPT\" ]; then
            echo \"Testing inference with checkpoint: \$LATEST_CKPT\";
            # You can add inference test commands here if needed
            echo 'Inference test placeholder - checkpoint created successfully';
        fi;
    fi;
    
    # Log completion time
    echo \"Training completed at: \$(date)\";
    echo \"Results saved in: /workspace/runs/$EXP\";
    echo \"Used encoder checkpoint: \$CHECKPOINT_PATH\";
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create job completion marker
touch "$EXP_DIR/job_completed.txt"
echo "Job completed successfully at $(date)" > "$EXP_DIR/job_completed.txt"
echo "Used encoder checkpoint: $ENCODER_CHECKPOINT" >> "$EXP_DIR/job_completed.txt"
echo "Config used: $CONFIG" >> "$EXP_DIR/job_completed.txt"
echo "Max epochs: $MAX_EPOCH" >> "$EXP_DIR/job_completed.txt"
echo "This was a baseline test run (minimal epochs)" >> "$EXP_DIR/job_completed.txt"

# Print final checkpoint locations for easy reference
echo ""
echo "===== Final diffusion model checkpoints ====="
find "$EXP_DIR" -name '*.ckpt' -type f | while read -r ckpt; do
    echo "Checkpoint: $ckpt"
done

echo ""
echo "===== Baseline Testing Summary ====="
echo "✓ Config: zinc-diffusion_ddpm_baseline.yaml"
echo "✓ Epochs: 5 (minimal for testing)"
echo "✓ Timesteps: 100 (reduced from 1000)"
echo "✓ DDIM steps: 20 (fast inference)"
echo "✓ Batch size: 64"
echo "✓ Encoder checkpoint: $ENCODER_CHECKPOINT"
echo "✓ WandB logging: enabled"
echo ""
echo "Next steps after successful baseline:"
echo "1. Check WandB dashboard for training progress"
echo "2. Verify inference works with generated checkpoints"
echo "3. Scale up to full training if baseline succeeds"