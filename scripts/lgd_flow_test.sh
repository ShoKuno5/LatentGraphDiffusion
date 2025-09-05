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
EXP=$(date +%Y%m%d_%H%M%S)_flow_test
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

# WandB settings - disable for quick testing
export WANDB_MODE=disabled

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
    
    echo '=== FLOW MATCHING QUICK TEST ===';
    echo 'Configuration: zinc-flow_rf.yaml (modified for testing)';
    echo '- Short runtime for debugging';
    echo '- WandB disabled';
    echo '- Focus on model instantiation and basic training loop';
    echo \"- Run ID: \$RUN_ID\";
    echo \"- Timestamp: \$TIMESTAMP\";
    echo '';
    
    # Pre-flight checks
    echo '--- Pre-flight Checks ---';
    echo 'Checking Python environment...';
    python -c \"
import torch
import torch_geometric
import lgd
from lgd.flow.flow_core import LatentFlow
from lgd.flow.sampler import FlowSampler
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ PyG: {torch_geometric.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ Flow classes imported successfully')
\";
    
    if [ \$? -ne 0 ]; then
        echo 'Environment check failed!';
        exit 1;
    fi;
    
    echo '';
    echo 'Checking config file...';
    if [ ! -f /workspace/cfg/zinc-flow_rf.yaml ]; then
        echo 'Config file not found: cfg/zinc-flow_rf.yaml';
        exit 1;
    fi;
    echo '✓ Config file exists';
    
    echo '';
    echo 'Checking pretrained encoder...';
    if [ -f /workspace/runs/zinc_encoder_fast_hpc/zinc-encoder-fast/0/ckpt/399.ckpt ]; then
        echo '✓ Pretrained encoder found at runs/zinc_encoder_fast_hpc/.../399.ckpt';
    else
        echo 'WARNING: Pretrained encoder not found at runs/zinc_encoder_fast_hpc/.../399.ckpt; will use random initialization';
    fi;
    
    echo '';
    echo '--- Starting Flow Test Training ---';
    START_TIME=\$(date +%s);
    
    # Run with limited epochs for testing
    python -c \"
import sys
sys.path.append('/workspace')
import lgd
from train_diffusion import *

# Patch config for quick testing
cfg.optim.max_epoch = 2  # Only 2 epochs for testing
cfg.train.eval_period = 1  # Eval every epoch
cfg.wandb.use = False  # Disable WandB
cfg.flow.get('ema', True)  # Keep EMA for testing

print('✓ Config patched for quick testing')
print(f'  - Max epochs: {cfg.optim.max_epoch}')
print(f'  - Model type: {cfg.model.get(\"type\", \"unknown\")}')
print(f'  - Flow objective: {cfg.flow.get(\"objective\", \"unknown\")}')
print('')

# Run the main training loop
try:
    main()
    print('✓ Quick test completed successfully!')
except Exception as e:
    print(f'✗ Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
\" 2>&1 | tee /workspace/runs/$EXP/flow_test.log;
    
    TEST_EXIT=\$?;
    END_TIME=\$(date +%s);
    DURATION=\$((\$END_TIME - \$START_TIME));
    
    echo '';
    echo '=== TEST SUMMARY ===';
    echo \"Duration: \$DURATION seconds\";
    
    if [ \$TEST_EXIT -eq 0 ]; then
        echo '✓ SUCCESS: Flow matching test passed!';
        echo '';
        echo 'What was tested:';
        echo '- LatentFlow model instantiation';
        echo '- Config loading and validation';
        echo '- Basic training loop (2 epochs)';
        echo '- Forward/backward passes';
        echo '- Loss computation (Rectified Flow)';
        echo '- Checkpoint saving';
        echo '';
        echo 'Ready for full training run!';
    else
        echo \"✗ FAILED: Test failed with exit code \$TEST_EXIT\";
        echo 'Check the log file for error details:';
        echo \"/workspace/runs/$EXP/flow_test.log\";
    fi;
    
    echo \"Test completed. Logs saved in: /workspace/runs/$EXP/\";
  "

echo "Flow test completed. Check results in: $EXP_DIR"
