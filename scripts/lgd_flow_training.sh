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
EXP=$(date +%Y%m%d_%H%M%S)_flow_training
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
    
    # 実行時にユニークなrun_idを生成
    RUN_ID=\$(date +%s);  # Unix timestamp
    TIMESTAMP=\$(date +%Y%m%d_%H%M%S);
    
    # 環境変数としてseedを設定
    export SEED=\$RUN_ID;
    
    echo 'Starting Latent Flow Matching training...';
    echo 'Configuration: zinc-flow_rf.yaml';
    echo '- Objective: Rectified Flow';
    echo '- ODE Solver: Heun (RK2)';
    echo '- NFE: 20 steps';
    echo '- Max epochs: 300';
    echo '- Pretrained encoder: results/zinc-encoder-fast/0/ckpt/19.ckpt';
    echo '- EMA enabled';
    echo '- WandB logging enabled';
    echo \"- Run ID: \$RUN_ID\";
    echo \"- Timestamp: \$TIMESTAMP\";
    echo \"- Seed (env): \$SEED\";
    echo \"- Results will be saved in: results/zinc-flow-rf/\$RUN_ID/\";
    echo '';
    echo 'NOTE: This is the second stage of LGD training (flow matching after encoder pretraining)';
    echo '';
    
    # Check if pretrained encoder exists
    if [ ! -f /workspace/results/zinc-encoder-fast/0/ckpt/19.ckpt ]; then
        echo 'WARNING: Pretrained encoder not found at results/zinc-encoder-fast/0/ckpt/19.ckpt';
        echo 'Please run encoder pretraining first or update the checkpoint path in zinc-flow_rf.yaml';
        echo '';
    fi;
    
    # Run the flow matching training with unique run_id via environment variable
    python train_diffusion.py --cfg cfg/zinc-flow_rf.yaml 2>&1 | tee /workspace/runs/$EXP/flow_training.log;
    
    # Check if training completed successfully
    if [ \$? -eq 0 ]; then
        echo 'Flow matching training completed successfully!';
        echo \"Results saved in: /workspace/runs/$EXP/\";
        echo \"Model checkpoints saved in: results/zinc-flow-rf/\$RUN_ID/\";
        
        # Find the best checkpoint in the run-specific directory
        BEST_CKPT=\$(find /workspace/results/zinc-flow-rf/\$RUN_ID -name \"*best*\" -type f | head -1);
        if [ ! -z \"\$BEST_CKPT\" ]; then
            echo \"Best checkpoint found: \$BEST_CKPT\";
        fi;
        
        # Summary of training stages
        echo '';
        echo '=== LGD Training Summary ===';
        echo 'Stage 1 (Encoder): results/zinc-encoder-fast/';
        echo \"Stage 2 (Flow): results/zinc-flow-rf/\$RUN_ID/\";
        echo 'Use the flow model checkpoint for graph generation via ODE sampling';
    else
        echo 'Flow training failed with exit code:' \$?;
        echo 'Check the log file for error details.';
    fi;
    
    echo 'Job completed. Check logs and WandB dashboard for results.';
  "

echo "Job completed. Results saved in: $EXP_DIR"
echo "Check WandB dashboard: https://wandb.ai/your_entity/LatentGraphDiffusion-ZINC-Flow"