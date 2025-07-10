# Latent Graph Diffusion (LGD) - Development Notes

## Environment Setup and Dependencies

### Key Findings from Analysis

1. **PyTorch and CUDA Versions**:
   - The `env.yaml` file specifies PyTorch 2.0.1 with CUDA 11.7
   - The existing `lgd_test.sh` script loads CUDA 12.6 module, which appears to be a mismatch
   - PyTorch Geometric packages showed compatibility issues in the logs due to undefined symbols

2. **Critical Dependencies**:
   - **Core ML**: PyTorch 2.0.1, PyTorch Geometric 2.3.1 (with torch-scatter, torch-sparse, torch-cluster, torch-spline-conv)
   - **Chemistry/Molecules**: RDKit 2023.9.2, ASE 3.23.0, OpenBabel, OGB 1.3.6
   - **ML Utilities**: WandB 0.18.7, PyTorch Lightning 2.1.2, Hydra 1.3.2
   - **Transformers**: Transformers 4.35.2, Performer-PyTorch 1.1.4
   - **Config Management**: YACS 0.1.8 (critical - was missing in initial attempts)

3. **PyG Compatibility Issues**:
   The error logs showed PyG extension libraries failing to load with undefined symbols. This is typically caused by:
   - Version mismatch between PyTorch and PyG packages
   - CUDA version incompatibility
   - Missing or incompatible C++ ABI versions

### Singularity Def File Design

The created `lgd.def` file addresses these issues by:

1. **Base Image**: Using `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` for consistency with env.yaml
2. **PyG Installation**: Installing from the correct wheel index for PyTorch 2.0.1 + CUDA 11.7
3. **Version Pinning**: Explicitly pinning PyG-related packages to compatible versions
4. **Comprehensive Dependencies**: Including all packages from env.yaml to ensure full compatibility

### Running on HPC

Based on `lgd_test.sh`, the typical workflow is:

```bash
# Build the Singularity image
singularity build lgd.sif lgd.def

# Run with proper bindings
singularity exec --nv \
  -B "$CODE":/workspace \
  -B "$RUNS":/workspace/runs \
  -B "$DATA":/workspace/data \
  lgd.sif \
  python pretrain.py --cfg cfg/zinc-encoder.yaml
```

### Environment Variables for HPC

Essential environment variables for distributed training:
- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=29500`
- `NCCL_IB_DISABLE=1`
- `NCCL_SOCKET_IFNAME=ib0,eth0`
- `GLOO_SOCKET_IFNAME=ib0,eth0`
- `OMP_NUM_THREADS=8`
- `WANDB_MODE=offline` (for offline logging)

### Common Issues and Solutions

1. **PyG Import Errors**: 
   - Ensure PyTorch and PyG versions match exactly
   - Use the correct CUDA version throughout

2. **Missing YACS Module**:
   - This was causing import failures - now included in def file

3. **CUDA Version Mismatch**:
   - The def file uses CUDA 11.7 to match PyTorch 2.0.1
   - If HPC loads different CUDA module, may need to adjust

### Testing the Environment

The def file includes a comprehensive test section that verifies:
- PyTorch and CUDA availability
- PyG extensions loading correctly
- Critical packages (ogb, wandb, rdkit, yacs) importing successfully

### Future Improvements

1. Consider upgrading to PyTorch 2.2+ with CUDA 12.1 for better performance
2. Add specific version pins for all pip packages to ensure reproducibility
3. Consider multi-stage build to reduce image size
4. Add support for custom PyG extensions if needed