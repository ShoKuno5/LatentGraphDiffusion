#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=04:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N zinc_diffusion_unconditional
#PJM -o zinc_diffusion_unconditional_%j.out
#PJM -e zinc_diffusion_unconditional_%j.err

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths --------
ROOT=/work/gp15/q25030
CODE=$ROOT/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- experiment tag ---------
EXP="zinc_diffusion_unconditional_$(date +%Y%m%d_%H%M%S)"
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
export WANDB_MODE=offline
export WANDB_PROJECT=LatentGraphDiffusion-ZINC
export WANDB_NAME="zinc_unconditional_${EXP}"

echo "Starting ZINC Unconditional Diffusion Training Job"
echo "Config: cfg/zinc-diffusion_ddpm_unconditional.yaml"
echo "Approach: UNCONDITIONAL generation (no conditioning)"
echo "Purpose: Fix inference assertion error from conditional mismatch"
echo "Encoder checkpoint: results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
echo "Epochs: 5 (baseline testing)"
echo "Experiment: $EXP"
echo "Time started: $(date)"
echo ""
echo "KEY CHANGES from conditional version:"
echo "- condition_list: ['unconditional']"
echo "- cond_stage_key: unconditional" 
echo "- Should resolve 'assert c is not None' error in inference"

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
    
    # Verify encoder checkpoint exists
    ENCODER_CKPT=\"results/zinc-encoder-fast/1754884349/ckpt/9.ckpt\";
    echo \"Checking encoder checkpoint: \$ENCODER_CKPT\";
    if [ ! -f \"\$ENCODER_CKPT\" ]; then
        echo 'ERROR: Encoder checkpoint file does not exist: '\$ENCODER_CKPT'';
        exit 1;
    fi;
    
    echo 'Starting ZINC UNCONDITIONAL diffusion training...';
    echo \"Configuration: cfg/zinc-diffusion_ddpm_unconditional.yaml\";
    echo \"- Training mode: train_diffusion\";
    echo \"- Max epochs: 5 (baseline)\";
    echo \"- Batch size: 64\";
    echo \"- Timesteps: 100\";
    echo \"- Encoder checkpoint: \$ENCODER_CKPT\";
    echo \"- Output directory: /workspace/runs/$EXP\";
    echo '';
    
    timeout 14400 python train_diffusion.py \\
        --cfg cfg/zinc-diffusion_ddpm_unconditional.yaml \\
        --repeat 1 \\
        wandb.use True \\
        wandb.entity shokuno-the-university-of-tokyo \\
        wandb.project LatentGraphDiffusion-ZINC \\
        wandb.name zinc_unconditional_${EXP} \\
        out_dir /workspace/runs/$EXP \\
        2>&1 | tee /workspace/runs/$EXP/training_unconditional.log;
    
    TRAIN_EXIT_CODE=\$?;
    
    # Check training results
    echo 'Training completed. Checking results...';
    find /workspace/runs/$EXP -name '*.ckpt' -exec ls -la {} \;
    
    echo \"Exit code: \$TRAIN_EXIT_CODE\";
    
    if [ \$TRAIN_EXIT_CODE -eq 0 ]; then
        echo 'SUCCESS: Unconditional diffusion training completed!';
        echo \"Checkpoints saved in: /workspace/runs/$EXP\";
        
        # Find the final checkpoint
        FINAL_CKPT=\$(find /workspace/runs/$EXP -name '*.ckpt' | sort | tail -1);
        if [ -n \"\$FINAL_CKPT\" ]; then
            echo \"Final checkpoint: \$FINAL_CKPT\";
            echo \"This checkpoint should work for inference without conditioning errors!\";
        fi;
    else
        echo 'FAILURE: Training failed!';
        echo \"Exit code: \$TRAIN_EXIT_CODE\";
        echo 'Check the logs for detailed error information.';
    fi;
    
    echo \"Training completed at: \$(date)\";
    echo \"Results directory: /workspace/runs/$EXP\";
    echo \"Encoder checkpoint used: \$ENCODER_CKPT\";
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create comprehensive completion report
touch "$EXP_DIR/job_completed.txt"
{
    echo "=== ZINC Unconditional Diffusion Training Job Completed ==="
    echo "Job completed at: $(date)"
    echo "Config used: cfg/zinc-diffusion_ddpm_unconditional.yaml"
    echo "Training approach: UNCONDITIONAL generation"
    echo "Purpose: Fix inference conditioning assertion error"
    echo "Encoder checkpoint: results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
    echo "Output directory: $EXP_DIR"
    echo ""
    echo "Key changes from conditional version:"
    echo "- condition_list: ['unconditional'] instead of ['masked_graph']"
    echo "- cond_stage_key: unconditional instead of masked_graph"
    echo ""
    echo "Expected outcome: Trained model compatible with inference-only mode"
    echo "Check training_unconditional.log for detailed results!"
} > "$EXP_DIR/job_completed.txt"

echo ""
echo "===== UNCONDITIONAL TRAINING SUMMARY ====="
echo "✓ Config: zinc-diffusion_ddmp_unconditional.yaml"
echo "✓ Training Mode: train_diffusion"
echo "✓ Generation Type: UNCONDITIONAL (no conditioning) 🎯"
echo "✓ Epochs: 5 (baseline testing)"
echo "✓ Purpose: Fix inference 'assert c is not None' error"
echo "✓ Encoder checkpoint: results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
echo ""
echo "🚀 This should train a model compatible with inference!"
echo "📈 Training time estimate: ~1-2 hours (similar to baseline)"
echo "🎯 Once complete, inference should work without conditioning errors"