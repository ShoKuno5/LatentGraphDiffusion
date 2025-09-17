#!/bin/bash
# Legacy Azure test: moved from scripts/

# Azure-compatible version of lgd_test.sh
# No HPC-specific directives, runs directly on Azure VM

# -------- paths --------
CODE=/home/azureuser/LatentGraphDiffusion
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
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8

# Check if WANDB_API_KEY is set, if not use offline mode
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_MODE=offline
    echo "WANDB_API_KEY not set, using offline mode"
else
    export WANDB_MODE=online
fi

export WANDB_PROJECT=latentgraphdiffusion
export WANDB_NAME="zinc_test_${EXP}"

# -------- Check if singularity is available --------
if command -v singularity &> /dev/null; then
    echo "Using Singularity container"
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
        python pretrain.py --cfg cfg/zinc-encoder.yaml --repeat 1 wandb.use False optim.max_epoch 1 train.ckpt_period 1 2>&1 | tee /workspace/runs/$EXP/pretrain_test.log;
        
        # Check if pretraining produced a checkpoint
        echo 'Searching for checkpoints...';
        echo 'Looking in /workspace/results/zinc-encoder/';
        find /workspace/results/zinc-encoder -name \"*.ckpt\" -ls 2>/dev/null || echo 'No checkpoints found in results/zinc-encoder';
        echo 'Looking in /workspace/runs/zinc-encoder/';
        find /workspace/runs/zinc-encoder -name \"*.ckpt\" -ls 2>/dev/null || echo 'No checkpoints found in runs/zinc-encoder';
        
        # Look in multiple possible locations for checkpoints
        CHECKPOINT_PATH=\$(find /workspace/results/zinc-encoder -name \"*.ckpt\" | head -1 2>/dev/null || find /workspace/runs/zinc-encoder -name \"*.ckpt\" | head -1 2>/dev/null || echo \"\");
        
        if [ -n \"\$CHECKPOINT_PATH\" ] && [ -f \"\$CHECKPOINT_PATH\" ]; then
            echo 'Autoencoder pretraining test completed successfully';
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
else
    echo "Singularity not found, running in local environment"
    # -------- Run directly in local Python environment --------
    cd "$CODE"
    export PYTHONPATH=$CODE:$PYTHONPATH
    export PYTHONUNBUFFERED=1
    
    # Create experiment directory
    mkdir -p "$RUNS/$EXP"
    
    echo 'Testing LGD environment setup...'
    python -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA device count: {torch.cuda.device_count()}")'
    
    echo 'Checking if required packages are installed...'
    python -c 'import torch_geometric; print(f"PyG version: {torch_geometric.__version__}")' || echo 'PyG not found'
    python -c 'import wandb; print(f"wandb version: {wandb.__version__}")' || echo 'wandb not found'
    python -c 'import ogb; print(f"OGB version: {ogb.__version__}")' || echo 'OGB not found'
    
    echo 'Testing LGD modules...'
    python -c 'from lgd.utils import *; print("LGD utils imported successfully")' || echo 'LGD utils import failed'
    
    echo 'Running LGD autoencoder pretraining test (short run)...'
    # First test: pretrain autoencoder with minimal settings
    python pretrain.py --cfg cfg/zinc-encoder.yaml --repeat 1 wandb.use False optim.max_epoch 1 train.ckpt_period 1 2>&1 | tee "$RUNS/$EXP/pretrain_test.log"
    
    # Check if pretraining produced a checkpoint
    echo 'Searching for checkpoints...'
    echo 'Looking in results/zinc-encoder/'
    find results/zinc-encoder -name "*.ckpt" -ls 2>/dev/null || echo 'No checkpoints found in results/zinc-encoder'
    echo 'Looking in runs/zinc-encoder/'
    find runs/zinc-encoder -name "*.ckpt" -ls 2>/dev/null || echo 'No checkpoints found in runs/zinc-encoder'
    
    # Look in multiple possible locations for checkpoints
    CHECKPOINT_PATH=$(find results/zinc-encoder -name "*.ckpt" | head -1 2>/dev/null || find runs/zinc-encoder -name "*.ckpt" | head -1 2>/dev/null || echo "")
    
    if [ -n "$CHECKPOINT_PATH" ] && [ -f "$CHECKPOINT_PATH" ]; then
        echo 'Autoencoder pretraining test completed successfully'
        echo "Found checkpoint: $CHECKPOINT_PATH"
        
        echo 'Running LGD diffusion training test (short run)...'
        # Second test: train diffusion model with minimal settings
        python train_diffusion.py --cfg cfg/zinc-diffusion_ddpm.yaml --repeat 1 wandb.use False optim.max_epoch 1 diffusion.first_stage_config "$CHECKPOINT_PATH" 2>&1 | tee "$RUNS/$EXP/diffusion_test.log"
        
        echo 'LGD diffusion training test completed'
    else
        echo 'Autoencoder pretraining failed - no checkpoint found'
    fi
    
    echo "LGD testing completed. Check logs in $RUNS/$EXP/"
fi

echo "Job completed. Results saved in: $EXP_DIR"
