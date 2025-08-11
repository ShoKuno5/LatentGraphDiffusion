#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=08:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths --------
# Use the currently open (gp15) workspace so edits are reflected inside the container
ROOT=/work/gp15/q25030
CODE=$ROOT/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- experiment tag ---------
EXP=$(date +%Y%m%d_%H%M%S)_fast_training
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

# WandB settings - will be loaded from .wandbrc file inside container
export WANDB_MODE=online

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
    
    echo 'Starting optimized LGD encoder pretraining...';
    echo 'Configuration: zinc-encoder-fast.yaml';
    echo '- Max epochs: 10 (down from 2000)';
    echo '- Hidden dimensions: 32 (down from 64)';  
    echo '- Early stopping enabled';
    echo '- WandB logging enabled';
    echo '';
    
    # Run the optimized training
    python pretrain.py --cfg cfg/zinc-encoder-fast.yaml 2>&1 | tee /workspace/runs/$EXP/pretrain_fast.log;
    
    # Check if training completed successfully
    if [ \$? -eq 0 ]; then
        echo 'Fast encoder pretraining completed successfully!';
        echo \"Results saved in: /workspace/runs/$EXP/\";
        
        # Find the best checkpoint
        BEST_CKPT=\$(find /workspace/runs -name \"*best*\" -type f | head -1);
        if [ ! -z \"\$BEST_CKPT\" ]; then
            echo \"Best checkpoint found: \$BEST_CKPT\";
        fi;
    else
        echo 'Training failed with exit code:' \$?;
    fi;
    
    echo 'Job completed. Check logs and WandB dashboard for results.';
  "

echo "Job completed. Results saved in: $EXP_DIR"
echo "Check WandB dashboard: https://wandb.ai/shokuno-the-university-of-tokyo/LatentGraphDiffusion-ZINC"