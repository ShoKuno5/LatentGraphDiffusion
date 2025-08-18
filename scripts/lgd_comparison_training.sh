#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=16:00:00
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
EXP=$(date +%Y%m%d_%H%M%S)_comparison_training
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
    
    echo '================================';
    echo 'LGD COMPARISON: DIFFUSION vs FLOW';
    echo '================================';
    echo \"Run ID: \$RUN_ID\";
    echo \"Timestamp: \$TIMESTAMP\";
    echo \"Seed (env): \$SEED\";
    echo '';
    echo 'This script will train both diffusion and flow matching models';
    echo 'for direct performance comparison on the same dataset.';
    echo '';
    
    # Check if pretrained encoder exists
    if [ ! -f /workspace/results/zinc-encoder-fast/0/ckpt/19.ckpt ]; then
        echo 'WARNING: Pretrained encoder not found at results/zinc-encoder-fast/0/ckpt/19.ckpt';
        echo 'Please run encoder pretraining first or update checkpoint paths in configs';
        echo '';
    fi;
    
    # Start timing
    START_TIME=\$(date +%s);
    
    echo '--- STAGE 1: DIFFUSION TRAINING ---';
    echo 'Configuration: zinc-diffusion_ddpm.yaml';
    echo 'Method: DDPM with 1000 timesteps';
    echo '';
    
    # Run diffusion training
    python train_diffusion.py --cfg cfg/zinc-diffusion_ddpm.yaml 2>&1 | tee /workspace/runs/$EXP/diffusion_training.log;
    DIFFUSION_EXIT=\$?;
    
    DIFF_END_TIME=\$(date +%s);
    DIFF_DURATION=\$((\$DIFF_END_TIME - \$START_TIME));
    
    if [ \$DIFFUSION_EXIT -eq 0 ]; then
        echo \"Diffusion training completed successfully in \$DIFF_DURATION seconds!\";
    else
        echo \"Diffusion training failed with exit code: \$DIFFUSION_EXIT\";
    fi;
    
    echo '';
    echo '--- STAGE 2: FLOW MATCHING TRAINING ---';
    echo 'Configuration: zinc-flow_rf.yaml';
    echo 'Method: Rectified Flow with 20 NFE';
    echo '';
    
    # Run flow matching training
    python train_diffusion.py --cfg cfg/zinc-flow_rf.yaml 2>&1 | tee /workspace/runs/$EXP/flow_training.log;
    FLOW_EXIT=\$?;
    
    END_TIME=\$(date +%s);
    FLOW_DURATION=\$((\$END_TIME - \$DIFF_END_TIME));
    TOTAL_DURATION=\$((\$END_TIME - \$START_TIME));
    
    if [ \$FLOW_EXIT -eq 0 ]; then
        echo \"Flow matching training completed successfully in \$FLOW_DURATION seconds!\";
    else
        echo \"Flow matching training failed with exit code: \$FLOW_EXIT\";
    fi;
    
    echo '';
    echo '=== COMPARISON SUMMARY ===';
    echo \"Total runtime: \$TOTAL_DURATION seconds\";
    echo \"Diffusion training: \$DIFF_DURATION seconds (exit: \$DIFFUSION_EXIT)\";
    echo \"Flow training: \$FLOW_DURATION seconds (exit: \$FLOW_EXIT)\";
    echo '';
    echo 'Results saved in:';
    if [ \$DIFFUSION_EXIT -eq 0 ]; then
        echo \"- Diffusion: results/zinc-diffusion_ddpm/\$RUN_ID/\";
    fi;
    if [ \$FLOW_EXIT -eq 0 ]; then
        echo \"- Flow: results/zinc-flow-rf/\$RUN_ID/\";
    fi;
    echo \"- Logs: /workspace/runs/$EXP/\";
    echo '';
    
    if [ \$DIFFUSION_EXIT -eq 0 ] && [ \$FLOW_EXIT -eq 0 ]; then
        echo 'SUCCESS: Both methods trained successfully!';
        echo 'Compare results via WandB dashboard or checkpoint evaluation.';
        echo '';
        echo 'For sampling comparison:';
        echo '1. Diffusion: Uses DDPM/DDIM with 1000→50 steps';
        echo '2. Flow: Uses ODE solver with 20 steps';
        echo '';
        echo 'Metrics to compare:';
        echo '- Training time per epoch';
        echo '- Sampling speed (NFE)';
        echo '- Generated molecule quality';
        echo '- Validation loss convergence';
    else
        echo 'One or both training runs failed. Check logs for details.';
    fi;
    
    echo 'Comparison job completed.';
  "

echo "Comparison job completed. Results saved in: $EXP_DIR"
echo "Check WandB dashboard for side-by-side comparison"