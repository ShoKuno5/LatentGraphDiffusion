# LGD Training Jobs for Wisteria HPC

This directory contains job scripts for training Latent Graph Diffusion models on the Wisteria HPC system using PJM (Platform Job Manager) and Singularity containers.

## Quick Start

1. **Make scripts executable:**
   ```bash
   chmod +x *.sh
   ```

2. **Submit a simple test job:**
   ```bash
   ./submit_jobs.sh test
   ```

3. **Train ZINC encoder:**
   ```bash
   ./submit_jobs.sh zinc-encoder
   ```

4. **Train ZINC diffusion model (after encoder is done):**
   ```bash
   ./submit_jobs.sh zinc-diffusion auto
   ```

## Available Job Scripts

### Core Training Scripts

- **`zinc_pretrain_encoder.sh`** - ZINC dataset encoder pretraining (48h, 2000 epochs)
- **`zinc_train_diffusion.sh`** - ZINC dataset diffusion model training (48h, 2000 epochs)  
- **`qm9_pretrain_encoder.sh`** - QM9 dataset encoder pretraining (72h, 1000 epochs)
- **`qm9_train_diffusion.sh`** - QM9 dataset diffusion model training (72h, 1000 epochs)
- **`general_pretrain_template.sh`** - Template for other datasets (physics, photo, etc.)
- **`lgd_test.sh`** - Quick environment test (2h, 1 epoch)

### Utility Scripts

- **`submit_jobs.sh`** - Job submission helper with validation
- **`monitor_jobs.sh`** - Job monitoring and status checking
- **`manage_checkpoints.sh`** - Checkpoint management utilities

## Usage Examples

### ZINC Dataset

```bash
# Submit encoder pretraining
./submit_jobs.sh zinc-encoder

# After encoder completes, submit diffusion training
./submit_jobs.sh zinc-diffusion auto

# Or specify specific checkpoint
./submit_jobs.sh zinc-diffusion /work/jh210022o/q25030/LatentGraphDiffusion/runs/zinc_encoder_20250713_120000/ckpt/best.ckpt
```

### QM9 Dataset

```bash
# Train encoder for dipole moment (mu)
./submit_jobs.sh qm9-encoder mu

# Train encoder for other properties
./submit_jobs.sh qm9-encoder alpha
./submit_jobs.sh qm9-encoder e_HOMO
./submit_jobs.sh qm9-encoder e_LUMO
./submit_jobs.sh qm9-encoder delta_e
./submit_jobs.sh qm9-encoder cv

# Train diffusion model
./submit_jobs.sh qm9-diffusion mu auto
```

### Other Datasets

```bash
# Physics dataset
./submit_jobs.sh general physics encoder
./submit_jobs.sh general physics diffusion auto

# Photo dataset  
./submit_jobs.sh general photo encoder
./submit_jobs.sh general photo diffusion /path/to/checkpoint.ckpt

# Node classification datasets
./submit_jobs.sh general ogbn-arxiv encoder
./submit_jobs.sh general cora encoder
```

## Monitoring Jobs

### Check job status:
```bash
# Basic job status
pjstat

# Detailed status
pjstat -v

# LGD-specific monitoring
./monitor_jobs.sh --status
```

### Monitor experiments:
```bash
# Show completed experiments
./monitor_jobs.sh --results

# Show recent logs
./monitor_jobs.sh --logs

# Continuous monitoring (refreshes every 30s)
./monitor_jobs.sh --watch
```

## Checkpoint Management

### List checkpoints:
```bash
# List all checkpoints
./manage_checkpoints.sh list

# Filter by dataset
./manage_checkpoints.sh list zinc
./manage_checkpoints.sh list qm9
```

### Find latest checkpoint:
```bash
# Find latest ZINC encoder checkpoint
./manage_checkpoints.sh find zinc encoder

# Find latest QM9 diffusion checkpoint
./manage_checkpoints.sh find qm9 diffusion
```

### Checkpoint info:
```bash
./manage_checkpoints.sh info /path/to/checkpoint.ckpt
```

### Backup and cleanup:
```bash
# Backup important checkpoint
./manage_checkpoints.sh backup /path/to/best_checkpoint.ckpt

# Clean old checkpoints (>30 days)
./manage_checkpoints.sh clean 30

# Create symbolic link
./manage_checkpoints.sh link /path/to/checkpoint.ckpt /path/to/link.ckpt
```

## Job Configuration

### Resource Requirements

| Job Type | Time Limit | Resources | Notes |
|----------|------------|-----------|-------|
| Test | 2h | 1 node | Quick validation |
| ZINC Encoder | 48h | 1 node | 2000 epochs |
| ZINC Diffusion | 48h | 1 node | 2000 epochs |
| QM9 Encoder | 72h | 1 node | 1000 epochs |
| QM9 Diffusion | 72h | 1 node | 1000 epochs |
| General | 48h | 1 node | Configurable |

### Environment Variables

All jobs use these environment settings:
- `WANDB_MODE=offline` - Offline logging
- `NCCL_IB_DISABLE=1` - Disable InfiniBand
- `OMP_NUM_THREADS=8` - OpenMP threads

### Data Paths

- **Code**: `/work/jh210022o/q25030/LatentGraphDiffusion`
- **Data**: `/work/jh210022o/q25030/LatentGraphDiffusion/data`
- **Results**: `/work/jh210022o/q25030/LatentGraphDiffusion/runs`
- **Container**: `/work/jh210022o/q25030/LatentGraphDiffusion/lgd.sif`

## Troubleshooting

### Common Issues

1. **Permission denied**: Make scripts executable with `chmod +x *.sh`

2. **Config file not found**: Check that the config file exists in `cfg/` directory

3. **Checkpoint not found**: Use `./manage_checkpoints.sh find` to locate checkpoints

4. **Job failed immediately**: Check the error output file `*_err` for details

5. **Out of time**: Increase time limit in job script (`#PJM -L elapse=XX:XX:XX`)

### Debug Commands

```bash
# Check job output
cat /work/jh210022o/q25030/LatentGraphDiffusion/experiments/wisteria/*_out

# Check job errors  
cat /work/jh210022o/q25030/LatentGraphDiffusion/experiments/wisteria/*_err

# Check experiment logs
ls -la /work/jh210022o/q25030/LatentGraphDiffusion/runs/*/

# Test environment
./submit_jobs.sh test
```

## Customization

### Modifying Job Parameters

Edit the job scripts to change:
- Time limits (`#PJM -L elapse=`)
- Resource requirements (`#PJM -L node=`)
- Training epochs (`MAX_EPOCH=`)
- Batch sizes (in config files)

### Adding New Datasets

1. Create config files in `cfg/` directory
2. Use `general_pretrain_template.sh` or create specific script
3. Update `submit_jobs.sh` if needed

## File Structure

```
experiments/wisteria/
├── README.md                      # This file
├── submit_jobs.sh                 # Job submission helper
├── monitor_jobs.sh                # Job monitoring
├── manage_checkpoints.sh          # Checkpoint management
├── lgd_test.sh                    # Environment test
├── zinc_pretrain_encoder.sh       # ZINC encoder training
├── zinc_train_diffusion.sh        # ZINC diffusion training
├── qm9_pretrain_encoder.sh        # QM9 encoder training
├── qm9_train_diffusion.sh         # QM9 diffusion training
└── general_pretrain_template.sh   # General template
```

## Support

For issues with:
- **Job scripts**: Check this README and script comments
- **LGD code**: See main repository documentation
- **HPC system**: Contact Wisteria support
- **Environment setup**: Check `lgd_test.sh` output