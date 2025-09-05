#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths --------
# Determine repo root (allow override via LGD_CODE)
if [ -n "$LGD_CODE" ]; then
  CODE="$LGD_CODE"
else
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
  CANDIDATES=(
    "$SCRIPT_DIR/.."
    "$PWD"
    "$PWD/.."
  )
  CODE=""
  for c in "${CANDIDATES[@]}"; do
    if [ -d "$c/cfg" ] && [ -d "$c/lgd" ]; then CODE="$c"; break; fi
  done
  if [ -z "$CODE" ]; then
    echo "ERROR: Could not locate repo root. Set LGD_CODE to your repo path." >&2
    exit 2
  fi
fi

# Allow override of Singularity image via LGD_IMG, else use repo-local image
IMG=${LGD_IMG:-"$CODE/lgd.sif"}
DATA=$CODE/data
RUNS=$CODE/runs

echo "Using CODE: $CODE"
echo "Using IMG : $IMG"

# Preflight: ensure the Singularity image exists
if [ ! -f "$IMG" ]; then
  echo "ERROR: Singularity image not found at: $IMG" >&2
  echo "Hint: Set LGD_IMG to your .sif path or place lgd.sif at repo root ($CODE)." >&2
  exit 2
fi

# -------- experiment tag ---------
EXP=$(date +%Y%m%d_%H%M%S)_flow_training
EXP_DIR=$RUNS/$EXP
mkdir -p "$DATA" "$EXP_DIR"
echo "Directory created: $EXP_DIR $DATA"

# Mirror all job stdout/stderr into the run directory as well
exec > >(tee -a "$EXP_DIR/pjm_stdout.log") 2>&1

# -------- env / NCCL / PyTorch --------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export GLOO_SOCKET_IFNAME=ib0,eth0
export OMP_NUM_THREADS=8

# WandB settings - read from .wandbrc file
export WANDB_MODE=online
export WANDB_PROJECT=LatentGraphDiffusion-ZINC-Flow
export WANDB_ENTITY=shokuno-the-university-of-tokyo

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
    
    # Read WandB API key from .wandbrc file
    if [ -f /workspace/.wandbrc ]; then
        export WANDB_API_KEY=\$(grep '^api_key' /workspace/.wandbrc | head -1 | cut -d'=' -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*\$//');
        echo \"WandB API key loaded from .wandbrc: \${WANDB_API_KEY:0:8}...\";
    else
        echo 'Warning: .wandbrc file not found, WandB may not work';
    fi;
    
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
    echo '- Max epochs: 10 (reduced for quick test)';
    echo '- Pretrained encoder: runs/zinc_encoder_fast_hpc/zinc-encoder-fast/0/ckpt/399.ckpt';
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
    if [ ! -f /workspace/runs/zinc_encoder_fast_hpc/zinc-encoder-fast/0/ckpt/399.ckpt ]; then
        echo 'WARNING: Pretrained encoder not found at runs/zinc_encoder_fast_hpc/zinc-encoder-fast/0/ckpt/399.ckpt';
        echo 'Please run encoder pretraining first or update flow.first_stage_config in zinc-flow_rf.yaml';
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
echo "Check WandB dashboard: https://wandb.ai/shokuno-the-university-of-tokyo/LatentGraphDiffusion-ZINC-Flow"
