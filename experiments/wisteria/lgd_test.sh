#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g jh210022a
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.6

# -------- host-side paths --------
ROOT=/work/jh210022o/q25030
CODE=$ROOT/LatentGraphDiffusion
IMG=$CODE/lgd.sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- experiment tag ---------
EXP=$(date +%Y%m%d_%H%M%S)          # ex.) 20250607_231045
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
export WANDB_NAME="zinc_test_${EXP}"

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
    
    echo 'Testing LGD environment setup...';
    python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA device count: {torch.cuda.device_count()}\")';
    
    echo 'Checking if required packages are installed...';
    python -c 'import torch_geometric; print(f\"PyG version: {torch_geometric.__version__}\")' || echo 'PyG not found';
    python -c 'import wandb; print(f\"wandb version: {wandb.__version__}\")' || echo 'wandb not found';
    python -c 'import ogb; print(f\"OGB version: {ogb.__version__}\")' || echo 'OGB not found';
    
    echo 'Testing LGD modules...';
    python -c 'from lgd.utils import *; print(\"LGD utils imported successfully\")' || echo 'LGD utils import failed';
    
    echo 'Running LGD autoencoder pretraining test (short run)...';
    # First test: pretrain autoencoder with minimal settings
    python pretrain.py --cfg cfg/zinc-encoder.yaml --repeat 1 wandb.use False optim.max_epoch 1 2>&1 | tee /workspace/runs/$EXP/pretrain_test.log;
    
    # Check if pretraining produced a checkpoint
    if [ -f /workspace/runs/zinc-encoder/*/checkpoints/epoch=0-step*.ckpt ]; then
        echo 'Autoencoder pretraining test completed successfully';
        
        # Update the diffusion config to use the generated checkpoint
        CHECKPOINT_PATH=\$(find /workspace/runs/zinc-encoder -name \"epoch=0-step*.ckpt\" | head -1);
        echo \"Found checkpoint: \$CHECKPOINT_PATH\";
        
        echo 'Running LGD diffusion training test (short run)...';
        # Second test: train diffusion model with minimal settings
        python train_diffusion.py --cfg cfg/zinc-diffusion_ddpm.yaml --repeat 1 wandb.use False optim.max_epoch 1 diffusion.first_stage_config \"\$CHECKPOINT_PATH\" 2>&1 | tee /workspace/runs/$EXP/diffusion_test.log;
        
        echo 'LGD diffusion training test completed';
    else
        echo 'Autoencoder pretraining failed - no checkpoint found';
    fi;
    
    echo 'LGD testing completed. Check logs in /workspace/runs/$EXP/';
  "

echo "Job completed. Results saved in: $EXP_DIR"
