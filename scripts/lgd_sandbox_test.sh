#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g jh210022a
#PJM -L jobenv=singularity
#PJM -j

source /etc/profile.d/modules.sh
module load singularity/3.7.3
module load cuda/12.0

# -------- host-side paths --------
ROOT=/work/jh210022o/q25030
CODE=$ROOT/LatentGraphDiffusion
IMG=$CODE/lgd_sandbox  # Use sandbox instead of .sif
DATA=$CODE/data
RUNS=$CODE/runs

# -------- experiment tag ---------
EXP=$(date +%Y%m%d_%H%M%S)_sandbox
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
export WANDB_NAME="zinc_sandbox_${EXP}"

# -------- singularity + LGD commands (sandbox mode) --------
singularity exec --nv --writable \
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
    
    echo 'Installing missing packages in sandbox...';
    # Install missing packages
    pip install yacs;
    pip install performer-pytorch;
    pip install tensorboardX;
    
    echo 'Testing LGD environment setup...';
    python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA device count: {torch.cuda.device_count()}\")';
    
    echo 'Checking if all required packages are now installed...';
    python -c 'import torch_geometric; print(f\"PyG version: {torch_geometric.__version__}\")';
    python -c 'import wandb; print(f\"wandb version: {wandb.__version__}\")';
    python -c 'import ogb; print(f\"OGB version: {ogb.__version__}\")';
    python -c 'import yacs; print(f\"yacs version: {yacs.__version__}\")';
    python -c 'import performer_pytorch; print(\"performer-pytorch imported successfully\")';
    python -c 'import tensorboardX; print(f\"tensorboardX version: {tensorboardX.__version__}\")';
    
    echo 'Testing LGD modules...';
    python -c 'from lgd.utils import *; print(\"LGD utils imported successfully\")';
    
    echo 'Running LGD autoencoder pretraining test (short run)...';
    # First test: pretrain autoencoder with minimal settings
    python pretrain.py --cfg cfg/zinc-encoder.yaml --repeat 1 wandb.use False model.max_epochs 1 2>&1 | tee /workspace/runs/$EXP/pretrain_test.log;
    
    # Check if pretraining produced a checkpoint
    if [ -f /workspace/runs/zinc-encoder/*/checkpoints/epoch=0-step*.ckpt ]; then
        echo 'Autoencoder pretraining test completed successfully';
        
        # Update the diffusion config to use the generated checkpoint
        CHECKPOINT_PATH=\$(find /workspace/runs/zinc-encoder -name \"epoch=0-step*.ckpt\" | head -1);
        echo \"Found checkpoint: \$CHECKPOINT_PATH\";
        
        echo 'Running LGD diffusion training test (short run)...';
        # Second test: train diffusion model with minimal settings
        python train_diffusion.py --cfg cfg/zinc-diffusion_ddpm.yaml --repeat 1 wandb.use False model.max_epochs 1 diffusion.first_stage_config \"\$CHECKPOINT_PATH\" 2>&1 | tee /workspace/runs/$EXP/diffusion_test.log;
        
        echo 'LGD diffusion training test completed';
    else
        echo 'Autoencoder pretraining failed - no checkpoint found';
    fi;
    
    echo 'LGD sandbox testing completed. Check logs in /workspace/runs/$EXP/';
    echo 'If everything works, you can build a new .sif with these packages installed.';
  "

echo "Sandbox job completed. Results saved in: $EXP_DIR"
