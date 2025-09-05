# Latent Graph Diffusion
Official Repository for NeurIPS 2024 Paper [Unifying Generation and Prediction on Graphs with Latent Graph Diffusion](https://openreview.net/pdf?id=lvibangnAs).

![Latent Graph Diffusion](./LatentGraphDiffusion.png)

In this paper, we propose the first framework that enables solving graph learning tasks of all levels (node, edge and graph) and all types (generation, regression and classification) using one formulation. We first formulate prediction tasks including regression and classification into a generic (conditional) generation framework, which enables diffusion models to perform deterministic tasks with provable guarantees. We then propose Latent Graph Diffusion (LGD), a generative model that can generate node, edge, and graph-level features of all categories simultaneously. We achieve this goal by embedding the graph structures and features into a latent space leveraging a powerful encoder and decoder, then training a diffusion model in the latent space. LGD is also capable of conditional generation through a specifically designed cross-attention mechanism. Leveraging LGD and the ``all tasks as generation'' formulation, our framework is capable of solving graph tasks of various levels and types. We verify the effectiveness of our framework with extensive experiments, where our models achieve state-of-the-art or highly competitive results across a wide range of generation and regression tasks.

### Python environment setup with Conda

We build our code based on [GraphGPS](https://github.com/rampasek/GraphGPS) with many modification and improvements, including combining it with DDPM.

```bash
conda create -n lgd python=3.9
conda activate lgd

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb

conda clean --all
```

### Running LGD

```bash
conda activate lgd

# An example to run experiments on Zinc dataset; change the configs files to run other experiments on different datasets with desired hyperparameters
# The commands are all in cfg/xxx.sh, and the configurations are set in cfg/xxx.yaml

# The first step is to pretrain an autoencoder
python pretrain.py --cfg cfg/zinc-encoder.yaml --repeat 5 wandb.use False

# Then train LGD
python train_diffusion.py --cfg cfg/zinc-diffusion_ddpm.yaml --repeat 5 wandb.use False

# Remember to change the file path of the checkpoint of the autoencoder in diffusion.first_stage_config

```

## Unified Operations Guide (HPC + Azure)

This project standardizes how we run training across environments. Use the
wrappers in `experiments/` which call reusable runners in `scripts/common/` and
source environment presets from `scripts/env/`.

### Repository Layout

- `cfg/`: YAML configs (datasets, diffusion, flow, unconditional)
- `scripts/env/`: environment presets (paths, NCCL/GLOO, WandB)
  - `wisteria.sh` (HPC), `azure.sh` (Azure VM)
- `scripts/common/`: inside-container runners (single source of truth)
  - ZINC: `run_zinc_pretrain_encoder.sh`, `run_zinc_train_diffusion.sh`,
    `run_zinc_train_flow.sh`, `run_zinc_train_diffusion_uncond.sh`
  - QM9: `run_qm9_pretrain_encoder.sh`, `run_qm9_train_diffusion.sh`
- `experiments/wisteria/`: PJM job wrappers
- `experiments/azure/`: Azure VM wrappers
- `scripts/legacy/`: older one-off scripts (kept for reference)

### WandB Configuration

Set `WANDB_API_KEY` and (optionally) `WANDB_ENTITY` in the environment, or
create a `.wandbrc` file at the repo root with:

```
api_key = <your-key>
entity  = <your-entity>
```

Wrappers automatically bind `.wandbrc` (if present) and export variables inside
the container.

### Wisteria HPC (PJM)

Submit jobs after making scripts executable (e.g., `chmod +x experiments/wisteria/*.sh`).

```
# ZINC encoder pretraining
pjsub experiments/wisteria/zinc_pretrain_encoder.sh

# ZINC diffusion (uses latest encoder checkpoint if not specified)
pjsub experiments/wisteria/zinc_train_diffusion.sh
pjsub experiments/wisteria/zinc_train_diffusion.sh /path/to/encoder.ckpt

# ZINC flow matching (Rectified Flow)
pjsub experiments/wisteria/zinc_train_flow.sh
pjsub experiments/wisteria/zinc_train_flow.sh /path/to/encoder.ckpt cfg/zinc-flow_rf.yaml 300

# ZINC unconditional diffusion
pjsub experiments/wisteria/zinc_train_diffusion_uncond.sh
pjsub experiments/wisteria/zinc_train_diffusion_uncond.sh /path/to/encoder.ckpt cfg/zinc-diffusion_ddpm_unconditional.yaml 5 50
```

### Azure VM

Run wrappers directly on the VM. If Singularity is available, training runs in
the `lgd.sif` container with proper bind mounts; otherwise it runs in the host
Python environment (ensure dependencies match `env.yaml`).

```
# ZINC encoder pretraining
experiments/azure/zinc_pretrain_encoder.sh
experiments/azure/zinc_pretrain_encoder.sh cfg/zinc-encoder.yaml 5 50

# ZINC diffusion (auto-detect encoder ckpt by default)
experiments/azure/zinc_train_diffusion.sh
experiments/azure/zinc_train_diffusion.sh auto cfg/zinc-diffusion_ddpm.yaml 5 50

# ZINC flow matching (Rectified Flow)
experiments/azure/zinc_train_flow.sh
experiments/azure/zinc_train_flow.sh /path/to/encoder.ckpt cfg/zinc-flow_rf.yaml 300

# ZINC unconditional diffusion
experiments/azure/zinc_train_diffusion_uncond.sh
experiments/azure/zinc_train_diffusion_uncond.sh auto cfg/zinc-diffusion_ddpm_unconditional.yaml 5 50
```

### Running Runners Directly (inside container)

For quick tests inside the container shell:

```
scripts/common/run_zinc_pretrain_encoder.sh --config cfg/zinc-encoder.yaml --repeat 1 --max-epoch 1
scripts/common/run_zinc_train_diffusion.sh --checkpoint auto --config cfg/zinc-diffusion_ddpm.yaml --repeat 1 --max-epoch 1
scripts/common/run_zinc_train_flow.sh --checkpoint auto --config cfg/zinc-flow_rf.yaml --max-epoch 10
scripts/common/run_zinc_train_diffusion_uncond.sh --checkpoint auto --config cfg/zinc-diffusion_ddpm_unconditional.yaml --repeat 1 --max-epoch 1
```

### Notes

- Checkpoints: runners auto-detect latest encoder ckpt when `--checkpoint auto`.
- Output: runners write logs/ckpts to `runs/<exp>`; YAML `out_dir` may also use `results/`.
- Legacy: older standalone scripts moved to `scripts/legacy/`.
