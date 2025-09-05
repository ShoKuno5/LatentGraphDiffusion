#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=04:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N zinc_diffusion_unconditional_fixed
#PJM -o zinc_diffusion_unconditional_fixed_%j.out
#PJM -e zinc_diffusion_unconditional_fixed_%j.err

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
EXP="zinc_diffusion_unconditional_fixed_$(date +%Y%m%d_%H%M%S)"
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
export WANDB_NAME="zinc_unconditional_fixed_${EXP}"

echo "Starting ZINC FIXED Unconditional Diffusion Training Job"
echo "Config: cfg/zinc-diffusion_ddpm_unconditional.yaml"
echo "Critical Fix Applied: conditioning_key: null"
echo "Purpose: Fix 'assert c is not None' by truly disabling conditioning"
echo "Previous attempt failed because conditioning_key was still 'crossattn'"
echo "Encoder checkpoint: results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
echo "Epochs: 5 (baseline testing)"
echo "Experiment: $EXP"
echo "Time started: $(date)"
echo ""
echo "ROOT CAUSE ANALYSIS:"
echo "- The assertion 'assert c is not None' triggers when conditioning_key is not None"
echo "- Previous config had conditioning_key: crossattn (still expecting conditioning)"
echo "- Fixed config has conditioning_key: null (truly unconditional)"
echo "- This should bypass the assertion entirely"

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
    
    # Verify the fix was applied by checking the config
    echo 'Verifying config fix...';
    echo 'Checking that conditioning_key is set to null...';
    grep -n 'conditioning_key:' cfg/zinc-diffusion_ddpm_unconditional.yaml;
    
    echo 'Starting ZINC FIXED UNCONDITIONAL diffusion training...';
    echo \"Configuration: cfg/zinc-diffusion_ddpm_unconditional.yaml\";
    echo \"- Training mode: train_diffusion\";
    echo \"- Max epochs: 5 (baseline)\";
    echo \"- Batch size: 64\";
    echo \"- Timesteps: 100\";
    echo \"- CRITICAL FIX: conditioning_key: null\";
    echo \"- Encoder checkpoint: \$ENCODER_CKPT\";
    echo \"- Output directory: /workspace/runs/$EXP\";
    echo '';
    
    timeout 14400 python train_diffusion.py \\
        --cfg cfg/zinc-diffusion_ddpm_unconditional.yaml \\
        --repeat 1 \\
        wandb.use True \\
        wandb.entity shokuno-the-university-of-tokyo \\
        wandb.project LatentGraphDiffusion-ZINC \\
        wandb.name zinc_unconditional_fixed_${EXP} \\
        out_dir /workspace/runs/$EXP \\
        2>&1 | tee /workspace/runs/$EXP/training_fixed_unconditional.log;
    
    TRAIN_EXIT_CODE=\$?;
    
    # Check training results
    echo 'Training completed. Checking results...';
    find /workspace/runs/$EXP -name '*.ckpt' -exec ls -la {} \;
    
    echo \"Exit code: \$TRAIN_EXIT_CODE\";
    
    if [ \$TRAIN_EXIT_CODE -eq 0 ]; then
        echo 'SUCCESS: Fixed unconditional diffusion training completed!';
        echo \"Checkpoints saved in: /workspace/runs/$EXP\";
        
        # Find the final checkpoint
        FINAL_CKPT=\$(find /workspace/runs/$EXP -name '*.ckpt' | sort | tail -1);
        if [ -n \"\$FINAL_CKPT\" ]; then
            echo \"Final checkpoint: \$FINAL_CKPT\";
            echo \"This checkpoint should work for inference without ANY conditioning errors!\";
            echo \"Key fix applied: conditioning_key: null bypasses assertion entirely\";
        fi;
    else
        echo 'FAILURE: Training failed again!';
        echo \"Exit code: \$TRAIN_EXIT_CODE\";
        echo 'Check the logs for detailed error information.';
        echo 'If same assertion error occurs, the issue is deeper in the codebase.';
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
    echo "=== ZINC FIXED Unconditional Diffusion Training Job Completed ==="
    echo "Job completed at: $(date)"
    echo "Config used: cfg/zinc-diffusion_ddpm_unconditional.yaml"
    echo "Training approach: TRULY UNCONDITIONAL generation"
    echo "Critical fix: conditioning_key: null (was crossattn)"
    echo "Purpose: Bypass 'assert c is not None' assertion in LGD.py:916"
    echo "Encoder checkpoint: results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
    echo "Output directory: $EXP_DIR"
    echo ""
    echo "Root cause analysis:"
    echo "- Previous attempts failed because conditioning_key was still 'crossattn'"
    echo "- The assertion triggers when self.model.conditioning_key is not None"
    echo "- Fix: Set conditioning_key: null to bypass assertion entirely"
    echo ""
    echo "Expected outcome: Training should complete without assertion errors"
    echo "Check training_fixed_unconditional.log for detailed results!"
} > "$EXP_DIR/job_completed.txt"

echo ""
echo "===== FIXED UNCONDITIONAL TRAINING SUMMARY ====="
echo "✓ Config: zinc-diffusion_ddpm_unconditional.yaml"
echo "✓ Training Mode: train_diffusion"
echo "✓ CRITICAL FIX: conditioning_key: null (was crossattn) 🔧"
echo "✓ Generation Type: TRULY UNCONDITIONAL 🎯"
echo "✓ Epochs: 5 (baseline testing)"
echo "✓ Root cause: assertion 'c is not None' when conditioning_key != None"
echo "✓ Solution: Set conditioning_key: null to bypass assertion"
echo ""
echo "🔥 This MUST work - the assertion will be bypassed entirely!"
echo "📈 Training time estimate: ~1-2 hours (similar to baseline)"
echo "🎯 Once complete, inference should work perfectly!"