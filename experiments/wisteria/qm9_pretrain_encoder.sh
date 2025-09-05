#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=72:00:00
#PJM -g gp15
#PJM -L jobenv=singularity
#PJM -j
#PJM -N qm9_pretrain_encoder
#PJM -o qm9_pretrain_encoder_%j.out
#PJM -e qm9_pretrain_encoder_%j.err

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths (via env file) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/env/wisteria.sh"

# -------- job parameters --------
# QM9 target property - options: mu, alpha, e_HOMO, e_LUMO, delta_e, cv
TARGET_PROPERTY="${1:-mu}"  # Default to mu (dipole moment)
CONFIG="cfg/QM9_regression_encoder_${TARGET_PROPERTY}.yaml"
REPEAT=3
MAX_EPOCH=50

# -------- experiment tag ---------
EXP="qm9_${TARGET_PROPERTY}_encoder_$(date +%Y%m%d_%H%M%S)"
EXP_DIR=$RUNS/$EXP
mkdir -p "$DATA" "$EXP_DIR"
echo "Directory created: $EXP_DIR $DATA"

# -------- env / NCCL / PyTorch --------
export WANDB_NAME="qm9_${TARGET_PROPERTY}_encoder_${EXP}"

echo "Starting QM9 Encoder Pretraining Job"
echo "Target Property: $TARGET_PROPERTY"
echo "Config: $CONFIG"
echo "Max Epochs: $MAX_EPOCH"
echo "Repeats: $REPEAT"
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
    
    /workspace/scripts/common/run_qm9_pretrain_encoder.sh \
      --target \"$TARGET_PROPERTY\" \
      --repeat \"$REPEAT\" \
      --max-epoch \"$MAX_EPOCH\" \
      --out-dir \"/workspace/runs/$EXP\" \
      --wandb-name \"qm9_${TARGET_PROPERTY}_encoder_${EXP}\";
    
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
echo "Target property: $TARGET_PROPERTY" >> "$EXP_DIR/job_completed.txt"
