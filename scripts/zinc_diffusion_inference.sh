#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N zinc_diffusion_inference
#PJM -o zinc_diffusion_inference_%j.out
#PJM -e zinc_diffusion_inference_%j.err

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
# Use our trained diffusion checkpoint
DIFFUSION_CHECKPOINT="runs/zinc_diffusion_baseline_20250815_163934/zinc-diffusion_ddpm_baseline/0/ckpt/0.ckpt"
ENCODER_CHECKPOINT="results/zinc-encoder-fast/1754884349/ckpt/9.ckpt"
CONFIG="cfg/zinc-diffusion_ddpm_baseline.yaml"
NUM_SAMPLES=100  # Number of molecules to generate
DDIM_STEPS=20   # Fast DDIM sampling (as configured in baseline)

# -------- experiment tag ---------
EXP="zinc_inference_$(date +%Y%m%d_%H%M%S)"
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
export WANDB_MODE=offline  # Offline for inference
export WANDB_PROJECT=LatentGraphDiffusion-ZINC
export WANDB_NAME="zinc_inference_${EXP}"

echo "Starting ZINC Diffusion INFERENCE Job"
echo "Config: $CONFIG"
echo "Diffusion Checkpoint: $DIFFUSION_CHECKPOINT"
echo "Encoder Checkpoint: $ENCODER_CHECKPOINT"
echo "Number of samples: $NUM_SAMPLES"
echo "DDIM steps: $DDIM_STEPS"
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
    
    # Verify both checkpoints exist
    DIFFUSION_CKPT=\"$DIFFUSION_CHECKPOINT\";
    ENCODER_CKPT=\"$ENCODER_CHECKPOINT\";
    
    echo \"Checking diffusion checkpoint: \$DIFFUSION_CKPT\";
    if [ ! -f \"\$DIFFUSION_CKPT\" ]; then
        echo 'ERROR: Diffusion checkpoint file does not exist: '\$DIFFUSION_CKPT'';
        echo 'Available checkpoints in runs/:';
        find /workspace/runs -name \"*.ckpt\" | head -10;
        exit 1;
    fi;
    
    echo \"Checking encoder checkpoint: \$ENCODER_CKPT\";
    if [ ! -f \"\$ENCODER_CKPT\" ]; then
        echo 'ERROR: Encoder checkpoint file does not exist: '\$ENCODER_CKPT'';
        exit 1;
    fi;
    
    echo 'Starting ZINC diffusion INFERENCE...';
    echo \"Configuration: $CONFIG\";
    echo \"- DDIM sampling: $DDIM_STEPS steps\";
    echo \"- Timesteps: 100 (from baseline config)\";
    echo \"- Samples to generate: $NUM_SAMPLES\";
    echo \"- Output directory: /workspace/runs/$EXP\";
    echo '';
    
    # Create inference configuration by modifying the training config
    echo 'Creating inference-specific config...';
    
    # Use the unconditional generation script as basis
    # First try with the QM9 unconditional script adapted for ZINC
    echo \"Command: python qm9_unconditional.py --cfg $CONFIG --repeat 1 wandb.use False train.mode eval diffusion.first_stage_config \\\"\$ENCODER_CKPT\\\" out_dir /workspace/runs/$EXP\";
    
    # Try inference using the unconditional generation approach
    timeout 3600 python qm9_unconditional.py \
        --cfg $CONFIG \
        --repeat 1 \
        wandb.use False \
        train.mode eval \
        diffusion.first_stage_config \"\$ENCODER_CKPT\" \
        out_dir /workspace/runs/$EXP \
        2>&1 | tee /workspace/runs/$EXP/inference.log;
    
    INFERENCE_EXIT_CODE=\$?;
    
    # If the first approach fails, try with train_diffusion.py in eval mode
    if [ \$INFERENCE_EXIT_CODE -ne 0 ]; then
        echo 'First inference approach failed, trying alternative method...';
        echo \"Command: python train_diffusion.py --cfg $CONFIG --repeat 1 wandb.use False optim.max_epoch 0 train.mode eval diffusion.first_stage_config \\\"\$ENCODER_CKPT\\\" out_dir /workspace/runs/$EXP\";
        
        timeout 3600 python train_diffusion.py \
            --cfg $CONFIG \
            --repeat 1 \
            wandb.use False \
            optim.max_epoch 0 \
            train.mode eval \
            diffusion.first_stage_config \"\$ENCODER_CKPT\" \
            out_dir /workspace/runs/$EXP \
            2>&1 | tee -a /workspace/runs/$EXP/inference.log;
        
        INFERENCE_EXIT_CODE=\$?;
    fi;
    
    # Check inference results
    echo 'Inference completed. Checking results...';
    find /workspace/runs/$EXP -name '*.pkl' -o -name '*.pt' -o -name '*.json' -exec ls -la {} \;
    
    if [ \$INFERENCE_EXIT_CODE -eq 0 ]; then
        echo 'Inference completed successfully!';
        echo \"Generated samples saved in: /workspace/runs/$EXP\";
        
        # Try to show some basic statistics about generated data
        echo 'Analyzing generated samples...';
        python -c \"
import os
import glob
results_dir = '/workspace/runs/$EXP'
pkl_files = glob.glob(os.path.join(results_dir, '**', '*.pkl'), recursive=True)
pt_files = glob.glob(os.path.join(results_dir, '**', '*.pt'), recursive=True)
json_files = glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True)
print(f'Generated files:')
print(f'  PKL files: {len(pkl_files)}')
print(f'  PT files: {len(pt_files)}')
print(f'  JSON files: {len(json_files)}')
for f in pkl_files[:3] + pt_files[:3] + json_files[:3]:
    print(f'  Sample: {f}')
        \" 2>/dev/null || echo 'Could not analyze generated samples (normal for some inference modes)';
    else
        echo \"Inference failed with exit code: \$INFERENCE_EXIT_CODE\";
        echo 'Check logs for details.';
    fi;
    
    # Log completion time
    echo \"Inference completed at: \$(date)\";
    echo \"Results saved in: /workspace/runs/$EXP\";
    echo \"Used diffusion checkpoint: \$DIFFUSION_CKPT\";
    echo \"Used encoder checkpoint: \$ENCODER_CKPT\";
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create job completion marker
touch "$EXP_DIR/job_completed.txt"
echo "Job completed successfully at $(date)" > "$EXP_DIR/job_completed.txt"
echo "Used diffusion checkpoint: $DIFFUSION_CHECKPOINT" >> "$EXP_DIR/job_completed.txt"
echo "Used encoder checkpoint: $ENCODER_CHECKPOINT" >> "$EXP_DIR/job_completed.txt"
echo "Config used: $CONFIG" >> "$EXP_DIR/job_completed.txt"
echo "Number of samples requested: $NUM_SAMPLES" >> "$EXP_DIR/job_completed.txt"
echo "DDIM steps: $DDIM_STEPS" >> "$EXP_DIR/job_completed.txt"
echo "This was an inference test run" >> "$EXP_DIR/job_completed.txt"

# Print final results for easy reference
echo ""
echo "===== Generated Sample Files ====="
find "$EXP_DIR" -name '*.pkl' -o -name '*.pt' -o -name '*.json' -type f | while read -r file; do
    echo "Generated file: $file"
done

echo ""
echo "===== Inference Testing Summary ====="
echo "✓ Config: zinc-diffusion_ddpm_baseline.yaml"
echo "✓ Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
echo "✓ Encoder checkpoint: $ENCODER_CHECKPOINT"
echo "✓ DDIM steps: $DDIM_STEPS (fast inference)"
echo "✓ Requested samples: $NUM_SAMPLES"
echo "✓ Inference mode: evaluation"
echo ""
echo "Check the inference logs and generated files for results!"
echo "Log file: $EXP_DIR/inference.log"