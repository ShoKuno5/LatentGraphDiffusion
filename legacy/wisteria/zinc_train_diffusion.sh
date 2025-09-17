#!/bin/bash
# Legacy wrapper: moved from experiments/wisteria/
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N zinc_train_diffusion
#PJM -o zinc_train_diffusion_%j.out
#PJM -e zinc_train_diffusion_%j.err

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths (via env file) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/ops/env/wisteria.sh"

# -------- job parameters --------
# Prefer environment variables (set via pjsub -x), fallback to positional args
# You can specify a checkpoint path, otherwise it will auto-detect
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-${1:-auto}}"  # e.g., pjsub -x ENCODER_CHECKPOINT=/path/to/ckpt
CONFIG="${CONFIG:-cfg/zinc-diffusion_ddpm.yaml}"
REPEAT="${REPEAT:-5}"
MAX_EPOCH="${MAX_EPOCH:-50}"

# -------- experiment tag ---------
EXP="zinc_diffusion_$(date +%Y%m%d_%H%M%S)"
EXP_DIR=$RUNS/$EXP
mkdir -p "$DATA" "$EXP_DIR"
echo "Directory created: $EXP_DIR $DATA"

# -------- env / NCCL / PyTorch --------
export WANDB_NAME="zinc_diffusion_${EXP}"

echo "Starting ZINC Diffusion Training Job"
echo "Config: $CONFIG"
echo "Max Epochs: $MAX_EPOCH"
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
    set -e;
    cd /workspace;
    export PYTHONPATH=/workspace:\$PYTHONPATH;
    export PYTHONUNBUFFERED=1;
    mkdir -p /workspace/runs/$EXP;
    /workspace/ops/runners/run_zinc_train_diffusion.sh \
      --checkpoint \"$ENCODER_CHECKPOINT\" \
      --config \"$CONFIG\" \
      --repeat \"$REPEAT\" \
      --max-epoch \"$MAX_EPOCH\" \
      --out-dir \"/workspace/runs/$EXP\" \
      --wandb-name \"zinc_diffusion_${EXP}\";
  "

echo "Job completed at: $(date)"
echo "Results saved in: $EXP_DIR"

# Create job completion marker
touch "$EXP_DIR/job_completed.txt"
{
  echo "Job completed successfully at $(date)";
  echo "Used encoder checkpoint: $ENCODER_CHECKPOINT";
  echo "Config: $CONFIG";
  echo "Repeat: $REPEAT, Max Epoch: $MAX_EPOCH";
} > "$EXP_DIR/job_completed.txt"
