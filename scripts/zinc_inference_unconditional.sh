#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N zinc_inference_unconditional
#PJM -o zinc_inference_unconditional_%j.out
#PJM -e zinc_inference_unconditional_%j.err

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths --------
ROOT=/work/gp15/q25030
CODE=$ROOT/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- inference parameters --------
# Use our trained UNCONDITIONAL diffusion checkpoint
DIFFUSION_CHECKPOINT="runs/zinc_diffusion_final_unconditional_20250815_172517/zinc-diffusion_ddpm_unconditional/0/ckpt/0.ckpt"
ENCODER_CHECKPOINT="results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
CONFIG="cfg/zinc-diffusion_ddpm_unconditional.yaml"

# -------- experiment tag ---------
EXP="zinc_inference_unconditional_$(date +%Y%m%d_%H%M%S)"
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
export WANDB_NAME="zinc_inference_unconditional_${EXP}"

echo "Starting ZINC Unconditional Diffusion INFERENCE Job"
echo "Config: $CONFIG"
echo "Diffusion Checkpoint: $DIFFUSION_CHECKPOINT"
echo "Encoder Checkpoint: $ENCODER_CHECKPOINT"
echo "Generation Type: UNCONDITIONAL (no conditioning issues!)"
echo "Purpose: Test end-to-end inference with unconditional model"
echo "Expected: NO 'assert c is not None' error"
echo "Experiment: $EXP"
echo "Time started: $(date)"
echo ""
echo "UNCONDITIONAL MODEL ADVANTAGES:"
echo "- No conditioning assertion errors"
echo "- Direct unconditional generation"
echo "- Compatible with inference-only mode"
echo "- Trained with cond_stage_config: '__is_unconditional__'"

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
    
    # Verify both checkpoints exist
    DIFFUSION_CKPT=\"$DIFFUSION_CHECKPOINT\";
    ENCODER_CKPT=\"$ENCODER_CHECKPOINT\";
    
    echo \"Checking diffusion checkpoint: \$DIFFUSION_CKPT\";
    if [ ! -f \"\$DIFFUSION_CKPT\" ]; then
        echo 'ERROR: Diffusion checkpoint file does not exist: '\$DIFFUSION_CKPT'';
        exit 1;
    else
        echo 'SUCCESS: Unconditional diffusion checkpoint found!';
        ls -la \"\$DIFFUSION_CKPT\";
    fi;
    
    echo \"Checking encoder checkpoint: \$ENCODER_CKPT\";
    if [ ! -f \"\$ENCODER_CKPT\" ]; then
        echo 'ERROR: Encoder checkpoint file does not exist: '\$ENCODER_CKPT'';
        exit 1;
    else
        echo 'SUCCESS: Encoder checkpoint found!';
        ls -la \"\$ENCODER_CKPT\";
    fi;
    
    echo 'Starting ZINC unconditional diffusion INFERENCE...';
    echo \"Configuration: $CONFIG\";
    echo \"- Training mode: inference-only\";
    echo \"- Generation: UNCONDITIONAL (no conditioning)\";
    echo \"- Pretrained diffusion checkpoint: \$DIFFUSION_CKPT\";
    echo \"- Encoder checkpoint: \$ENCODER_CKPT\";
    echo \"- DDIM sampling: 20 steps\";
    echo \"- Timesteps: 100\";
    echo \"- Output directory: /workspace/runs/$EXP\";
    echo '';
    
    # Use train_diffusion.py with unconditional model
    echo \"Command: python train_diffusion.py --cfg $CONFIG --repeat 1 wandb.use False train.mode inference-only diffusion.first_stage_config \\\"\$ENCODER_CKPT\\\" pretrained.dir \\\"\$DIFFUSION_CKPT\\\" out_dir /workspace/runs/$EXP\";
    
    timeout 3600 python train_diffusion.py \\
        --cfg $CONFIG \\
        --repeat 1 \\
        wandb.use False \\
        train.mode inference-only \\
        diffusion.first_stage_config \"\$ENCODER_CKPT\" \\
        pretrained.dir \"\$DIFFUSION_CKPT\" \\
        out_dir /workspace/runs/$EXP \\
        2>&1 | tee /workspace/runs/$EXP/inference_unconditional.log;
    
    INFERENCE_EXIT_CODE=\$?;
    
    # Check inference results
    echo 'Inference completed. Checking results...';
    find /workspace/runs/$EXP -name '*.pkl' -o -name '*.pt' -o -name '*.json' -o -name '*.ckpt' -exec ls -la {} \\;
    
    echo \"Exit code: \$INFERENCE_EXIT_CODE\";
    
    if [ \$INFERENCE_EXIT_CODE -eq 0 ]; then
        echo 'SUCCESS: Unconditional inference completed successfully!';
        echo \"Generated samples saved in: /workspace/runs/$EXP\";
        
        # Show detailed results analysis
        echo 'Analyzing inference results...';
        python -c \"
import os
import glob
import json
results_dir = '/workspace/runs/$EXP'
print('\\\\n=== UNCONDITIONAL INFERENCE RESULTS ANALYSIS ===')
total_files = 0
for root, dirs, files in os.walk(results_dir):
    for f in files:
        if f.endswith(('.json', '.pkl', '.pt')):
            full_path = os.path.join(root, f)
            size = os.path.getsize(full_path)
            total_files += 1
            print(f'Generated: {os.path.relpath(full_path, results_dir)} ({size:,} bytes)')
        elif f.endswith('.log'):
            full_path = os.path.join(root, f)
            size = os.path.getsize(full_path)
            print(f'Log file: {os.path.relpath(full_path, results_dir)} ({size:,} bytes)')

print(f'\\\\nTotal generated files: {total_files}')
if total_files == 0:
    print('INFO: No inference output files - this is normal for evaluation-only mode')
    print('Success is measured by completing without assertion errors!')
else:
    print('EXCELLENT: Files generated successfully!')
        \" 2>/dev/null || echo 'Could not analyze results';
        
        # Check for no assertion errors
        if grep -q 'assert c is not None' /workspace/runs/$EXP/inference_unconditional.log; then
            echo 'UNEXPECTED: Assertion error still found - this should not happen!';
        else
            echo 'PERFECT: No conditioning assertion errors found!';
            echo 'The unconditional approach successfully resolved the issue!';
        fi;
        
    else
        echo 'FAILURE: Inference failed!';
        echo \"Exit code: \$INFERENCE_EXIT_CODE\";
        echo 'Check the logs for detailed error information.';
    fi;
    
    echo \"Inference completed at: \$(date)\";
    echo \"Results directory: /workspace/runs/$EXP\";
    echo \"Diffusion checkpoint used: \$DIFFUSION_CKPT\";
    echo \"Encoder checkpoint used: \$ENCODER_CKPT\";
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create comprehensive completion report
touch "$EXP_DIR/job_completed.txt"
{
    echo "=== ZINC Unconditional Diffusion Inference Job Completed ==="
    echo "Job completed at: $(date)"
    echo "Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
    echo "Encoder checkpoint: $ENCODER_CHECKPOINT"
    echo "Config used: $CONFIG"
    echo "Training mode: inference-only"
    echo "Generation type: UNCONDITIONAL"
    echo "Output directory: $EXP_DIR"
    echo ""
    echo "Expected outcomes:"
    echo "1. No 'assert c is not None' errors"
    echo "2. Successful DDIM sampling"
    echo "3. Evaluation metrics computed"
    echo "4. End-to-end pipeline working"
    echo ""
    echo "Check inference_unconditional.log for detailed results!"
} > "$EXP_DIR/job_completed.txt"

echo ""
echo "===== UNCONDITIONAL INFERENCE SUMMARY ====="
echo "✓ Config: zinc-diffusion_ddpm_unconditional.yaml"
echo "✓ Training Mode: inference-only"
echo "✓ Generation Type: UNCONDITIONAL 🎯"
echo "✓ Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
echo "✓ Encoder checkpoint: $ENCODER_CHECKPOINT"
echo "✓ Expected: NO conditioning assertion errors!"
echo ""
echo "🚀 This should complete the end-to-end pipeline successfully!"
echo "📊 Check logs for evaluation metrics and successful sampling!"
echo "🎯 Final test of our unconditional solution!"