#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=00:30:00
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
EXP=$(date +%Y%m%d_%H%M%S)_flow_debug
EXP_DIR=$RUNS/$EXP
mkdir -p "$DATA" "$EXP_DIR"
echo "Directory created: $EXP_DIR"

# Mirror all job stdout/stderr into the run directory as well
exec > >(tee -a "$EXP_DIR/pjm_stdout.log") 2>&1

# -------- env / NCCL / PyTorch --------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export GLOO_SOCKET_IFNAME=ib0,eth0
export OMP_NUM_THREADS=8
export WANDB_MODE=disabled

# -------- singularity debug test --------
echo "Starting Flow Matching Debug Test..."
echo "Output directory: $EXP_DIR"
echo ""

singularity exec --nv \
  -B "$CODE":/workspace \
  -B "$RUNS":/workspace/runs \
  -B "$DATA":/workspace/data \
  "$IMG" \
  bash -c "
    cd /workspace;
    export PYTHONPATH=/workspace:\$PYTHONPATH;
    export PYTHONUNBUFFERED=1;
    
    mkdir -p /workspace/runs/$EXP;
    
    echo '=== FLOW MATCHING DEBUG TEST ===';
    echo 'Timestamp: '\$(date);
    echo '';
    
    # Test 1: Python environment
    echo '--- Test 1: Python Environment ---';
    python -c \"
import sys
print(f'Python: {sys.version}')
print(f'Python path: {sys.path[0]}')
    \";
    
    # Test 2: PyTorch and CUDA
    echo '';
    echo '--- Test 2: PyTorch and CUDA ---';
    python -c \"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    \";
    
    # Test 3: PyTorch Geometric
    echo '';
    echo '--- Test 3: PyTorch Geometric ---';
    python -c \"
try:
    import torch_geometric
    print(f'PyG version: {torch_geometric.__version__}')
    from torch_geometric.graphgym.config import cfg
    print('GraphGym config: OK')
except Exception as e:
    print(f'ERROR: {e}')
    \";
    
    # Test 4: LGD imports
    echo '';
    echo '--- Test 4: LGD Core Imports ---';
    python -c \"
import sys
sys.path.insert(0, '/workspace')
try:
    import lgd
    print('lgd package: OK')
    from lgd.model.GraphTransformerEncoder import GraphTransformerEncoder
    print('GraphTransformerEncoder: OK')
    from lgd.model.DenoisingTransformer import DenoisingTransformer
    print('DenoisingTransformer: OK')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
    \";
    
    # Test 5: Flow module imports
    echo '';
    echo '--- Test 5: Flow Module Imports ---';
    python -c \"
import sys
sys.path.insert(0, '/workspace')
try:
    from lgd.flow.flow_core import LatentFlow, VelocityWrapper
    print('LatentFlow: OK')
    from lgd.flow.sampler import FlowSampler, solve_flow
    print('FlowSampler: OK')
    print('✓ All flow modules imported successfully!')
except Exception as e:
    print(f'ERROR importing flow modules: {e}')
    import traceback
    traceback.print_exc()
    \";
    
    # Test 6: Config loading
    echo '';
    echo '--- Test 6: Config Loading ---';
    python -c \"
import yaml
import os

config_path = 'cfg/zinc-flow_rf.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Config loaded from: {config_path}')
    print(f'Model type: {config.get(\"model\", {}).get(\"type\")}')
    print(f'Flow objective: {config.get(\"flow\", {}).get(\"objective\")}')
    encoder_path = config.get(\"flow\", {}).get(\"first_stage_config\")
    print(f'Encoder path: {encoder_path}')
    if os.path.exists(encoder_path):
        print(f'✓ Encoder checkpoint exists')
    else:
        print(f'✗ Encoder checkpoint NOT found')
else:
    print(f'✗ Config file not found: {config_path}')
    \";
    
    # Test 7: Model instantiation
    echo '';
    echo '--- Test 7: Model Instantiation ---';
    python -c \"
import sys
import os
import torch
sys.path.insert(0, '/workspace')

try:
    from torch_geometric.graphgym.config import cfg, set_cfg
    from lgd.flow.flow_core import LatentFlow
    
    # Minimal setup
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config sections
    import yaml
    with open('cfg/zinc-flow_rf.yaml', 'r') as f:
        flow_config = yaml.safe_load(f)
    
    cfg.flow = flow_config.get('flow', {})
    cfg.encoder = flow_config.get('encoder', {})
    cfg.dt = flow_config.get('dt', {})
    
    # Check encoder path
    encoder_path = cfg.flow.get('first_stage_config')
    if not os.path.exists(encoder_path):
        print(f'Creating dummy encoder for test...')
        # Create dummy checkpoint
        torch.save({'state_dict': {}, 'model_state': {}}, 'dummy_encoder.ckpt')
        encoder_path = 'dummy_encoder.ckpt'
    
    # Try to create model
    print('Creating LatentFlow model...')
    model = LatentFlow(
        first_stage_config=encoder_path,
        objective=cfg.flow.get('objective', 'rectified'),
        cond_stage_config='__is_unconditional__',
        hid_dim=cfg.flow.get('hid_dim', 4),
        use_ema=False
    )
    
    print('✓ Model created successfully!')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
except Exception as e:
    print(f'✗ Model creation failed: {e}')
    import traceback
    traceback.print_exc()
    \";
    
    # Test 8: Simple forward pass
    echo '';
    echo '--- Test 8: Simple Forward Pass ---';
    python /workspace/test_flow.py 2>&1 | tee /workspace/runs/$EXP/test_output.log;
    
    echo '';
    echo '=== DEBUG TEST COMPLETE ===';
    echo 'Check logs in: /workspace/runs/$EXP/';
    
    # Final summary
    if [ -f /workspace/runs/$EXP/test_output.log ]; then
        echo '';
        echo 'Test output saved. Checking for errors...';
        if grep -q 'ALL TESTS PASSED' /workspace/runs/$EXP/test_output.log; then
            echo '✓ All tests passed!';
            exit 0;
        else
            echo '✗ Some tests failed. Check the log for details.';
            exit 1;
        fi;
    fi;
  " 2>&1 | tee "$EXP_DIR/debug.log"

EXIT_CODE=$?
echo ""
echo "Debug test completed with exit code: $EXIT_CODE"
echo "Full log saved in: $EXP_DIR/debug.log"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Flow matching is ready for training!"
    echo "Next step: pjsub scripts/lgd_flow_training.sh"
else
    echo "✗ Issues found. Please review the debug log."
fi
