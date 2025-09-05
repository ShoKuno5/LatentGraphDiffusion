# LGD Training Jobs for Wisteria HPC

Please see the unified guide in the repository root `README.md` for complete
instructions covering Wisteria and Azure usage. This directory contains the
PJM job wrappers referenced there. For convenience, common commands are:

```bash
# Make scripts executable
chmod +x *.sh

# Submit encoder pretraining
pjsub zinc_pretrain_encoder.sh

# Diffusion (auto-detects latest encoder ckpt)
pjsub zinc_train_diffusion.sh
pjsub zinc_train_diffusion.sh /path/to/encoder.ckpt

# Flow matching (Rectified Flow)
pjsub zinc_train_flow.sh
pjsub zinc_train_flow.sh /path/to/encoder.ckpt cfg/zinc-flow_rf.yaml 300

# Unconditional diffusion
pjsub zinc_train_diffusion_uncond.sh
pjsub zinc_train_diffusion_uncond.sh /path/to/encoder.ckpt cfg/zinc-diffusion_ddpm_unconditional.yaml 5 50
```
