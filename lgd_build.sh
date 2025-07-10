#!/bin/bash
# Build script for LGD Singularity image on HPC

# First, build the SIF image
echo "Building SIF image..."
singularity build --fakeroot lgd.sif lgd.def

# Test the SIF image
echo "Testing SIF image..."
singularity exec lgd.sif python -c "import torch; print('PyTorch:', torch.__version__)"

# If you need a sandbox, try without fakeroot first
echo "Creating sandbox from SIF..."
singularity build --sandbox lgd_sandbox lgd.sif

# Alternative: use the SIF directly without sandbox
echo "You can use the SIF image directly with:"
echo "singularity exec --nv lgd.sif bash"
echo "or"
echo "singularity shell --nv lgd.sif"