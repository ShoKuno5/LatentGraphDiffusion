#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g jh210022a
#PJM -L jobenv=singularity
#PJM -j
#PJM -N general_pretrain
#PJM -o general_pretrain_%j.out
#PJM -e general_pretrain_%j.err

# Usage: ./general_pretrain_template.sh <dataset> <task_type>
# Examples:
#   ./general_pretrain_template.sh physics encoder
#   ./general_pretrain_template.sh photo diffusion /path/to/encoder/checkpoint.ckpt

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- job parameters --------
DATASET="${1:-physics}"      # Dataset name (physics, photo, ogbn-arxiv, etc.)
TASK_TYPE="${2:-encoder}"    # Task type: encoder or diffusion
ENCODER_CHECKPOINT="${3:-auto}"  # For diffusion jobs, path to encoder checkpoint

# Determine config file based on dataset and task
if [ "$TASK_TYPE" = "encoder" ]; then
    CONFIG="cfg/${DATASET}-encoder.yaml"
elif [ "$TASK_TYPE" = "diffusion" ]; then
    CONFIG="cfg/${DATASET}-diffusion.yaml"
else
    echo "ERROR: Invalid task type. Use 'encoder' or 'diffusion'"
    exit 1
fi

# -------- host-side paths --------
ROOT=/work/jh210022o/q25030
CODE=$ROOT/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- experiment parameters --------
REPEAT=3
if [ "$DATASET" = "ogbn-arxiv" ] || [ "$DATASET" = "cora" ] || [ "$DATASET" = "pubmed" ]; then
    MAX_EPOCH=50  # Node classification tasks
else
    MAX_EPOCH=50  # Graph classification tasks
fi

# -------- experiment tag ---------
EXP="${DATASET}_${TASK_TYPE}_$(date +%Y%m%d_%H%M%S)"
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
export WANDB_API_KEY=fb39ca5f5835abaa4c40a8b61dde2a499b45fcba
export WANDB_PROJECT=latentgraphdiffusion
export WANDB_NAME="${DATASET}_${TASK_TYPE}_${EXP}"

echo "Starting General Training Job"
echo "Dataset: $DATASET"
echo "Task Type: $TASK_TYPE"
echo "Config: $CONFIG"
echo "Max Epochs: $MAX_EPOCH"
echo "Repeats: $REPEAT"
if [ "$TASK_TYPE" = "diffusion" ]; then
    echo "Encoder Checkpoint: $ENCODER_CHECKPOINT"
fi
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
    
    # Verify config file exists
    if [ ! -f \"$CONFIG\" ]; then
        echo 'ERROR: Config file does not exist: $CONFIG';
        echo 'Available configs for $DATASET:';
        ls -la cfg/${DATASET}*.yaml;
        exit 1;
    fi;
    
    if [ \"$TASK_TYPE\" = \"encoder\" ]; then
        echo \"Starting $DATASET encoder pretraining...\";
        python pretrain.py \
            --cfg $CONFIG \
            --repeat $REPEAT \
            wandb.use True \
            optim.max_epoch $MAX_EPOCH \
            out_dir /workspace/runs/$EXP \
            wandb.name \"${DATASET}_${TASK_TYPE}_${EXP}\" \
            2>&1 | tee /workspace/runs/$EXP/pretrain_${DATASET}.log;
    
    elif [ \"$TASK_TYPE\" = \"diffusion\" ]; then
        # Auto-detect checkpoint if needed
        if [ \"$ENCODER_CHECKPOINT\" = \"auto\" ]; then
            echo 'Auto-detecting latest $DATASET encoder checkpoint...';
            CHECKPOINT_PATH=\$(find /workspace/runs -path \"*/${DATASET}_encoder_*/ckpt/*.ckpt\" | sort -V | tail -1);
            if [ -z \"\$CHECKPOINT_PATH\" ]; then
                echo 'ERROR: No $DATASET encoder checkpoint found. Please run encoder pretraining first or specify checkpoint path.';
                echo 'Available checkpoints:';
                find /workspace/runs -name \"*.ckpt\" | grep -i $DATASET | head -10;
                exit 1;
            fi;
        else
            CHECKPOINT_PATH=\"$ENCODER_CHECKPOINT\";
        fi;
        
        echo \"Using encoder checkpoint: \$CHECKPOINT_PATH\";
        
        # Verify checkpoint exists
        if [ ! -f \"\$CHECKPOINT_PATH\" ]; then
            echo 'ERROR: Checkpoint file does not exist: '\$CHECKPOINT_PATH'';
            exit 1;
        fi;
        
        echo \"Starting $DATASET diffusion training...\";
        python train_diffusion.py \
            --cfg $CONFIG \
            --repeat $REPEAT \
            wandb.use True \
            optim.max_epoch $MAX_EPOCH \
            diffusion.first_stage_config \"\$CHECKPOINT_PATH\" \
            out_dir /workspace/runs/$EXP \
            wandb.name \"${DATASET}_${TASK_TYPE}_${EXP}\" \
            2>&1 | tee /workspace/runs/$EXP/diffusion_${DATASET}.log;
    fi;
    
    # Check training results
    echo 'Training completed. Checking results...';
    find /workspace/runs/$EXP -name '*.ckpt' -exec ls -la {} \;
    
    # Log completion time
    echo \"Training completed at: \$(date)\";
    echo \"Results saved in: /workspace/runs/$EXP\";
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create job completion marker
touch "$EXP_DIR/job_completed.txt"
echo "Job completed successfully at $(date)" > "$EXP_DIR/job_completed.txt"
echo "Dataset: $DATASET" >> "$EXP_DIR/job_completed.txt"
echo "Task Type: $TASK_TYPE" >> "$EXP_DIR/job_completed.txt"
if [ "$TASK_TYPE" = "diffusion" ]; then
    echo "Used encoder checkpoint: $ENCODER_CHECKPOINT" >> "$EXP_DIR/job_completed.txt"
fi